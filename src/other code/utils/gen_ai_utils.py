import requests
import time
import google.auth
import google.auth.transport.requests
from google.cloud import bigquery


def get_auth_headers():
    """Get authentication headers for Google Cloud API calls."""
    creds, _ = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    access_token = creds.token

    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }


def make_api_request(url: str, method: str, data: dict = None):
    """Make an API request with proper authentication."""
    headers = get_auth_headers()

    if method == "GET":
        response = requests.get(url, headers=headers)
    elif method == "POST":
        response = requests.post(url, json=data, headers=headers)
    elif method == "PUT":
        response = requests.put(url, json=data, headers=headers)
    elif method == "PATCH":
        response = requests.patch(url, json=data, headers=headers)
    elif method == "DELETE":
        response = requests.delete(url, headers=headers)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")

    if response.status_code == 200:
        return response.json()
    else:
        error = (
            f"API request failed -> Status: {response.status_code} "
            f"Text: {response.text}"
        )
        raise RuntimeError(error)


def create_vertex_ai_connection(params):
    """Creates a Vertex AI connection."""
    project_id = params["project_id"]
    bigquery_location = params["bigquery_location"]
    vertex_ai_connection_name = params["vertex_ai_connection_name"]

    base_url = "https://bigqueryconnection.googleapis.com/v1"
    connections_path = (
        f"projects/{project_id}/locations/{bigquery_location}/connections"
    )
    url = f"{base_url}/{connections_path}"

    # Check if connection exists
    try:
        json_result = make_api_request(url, "GET")
        print(f"Checking existing connections...")

        if "connections" in json_result:
            for item in json_result["connections"]:
                print(f"Found connection: {item['name']}")
                conn_path = (
                    f"/locations/{bigquery_location}/connections/"
                    f"{vertex_ai_connection_name}"
                )
                if item["name"].endswith(conn_path):
                    print("Connection already exists")
                    return item["cloudResource"]["serviceAccountId"]
    except Exception as e:
        print(f"Error checking connections: {str(e)}")
        raise

    # Create new connection
    print("Creating new Vertex AI Connection...")
    create_url = f"{url}?connectionId={vertex_ai_connection_name}"

    request_body = {
        "friendlyName": "notebook_connection",
        "description": "Vertex AI Notebooks Connection for Data Analytics",
        "cloudResource": {},
    }

    try:
        json_result = make_api_request(create_url, "POST", request_body)
        service_account_id = json_result["cloudResource"]["serviceAccountId"]
        print("Vertex AI Connection created:", service_account_id)
        return service_account_id
    except Exception as e:
        print(f"Error creating connection: {str(e)}")
        raise


def set_project_level_iam_policy(params, accountWithPrefix, role):
    """Sets the Project Level IAM policy."""

    # Get the current bindings (if the account has access then skip)
    # https://cloud.google.com/resource-manager/reference/rest/v1/projects/getIamPolicy
    project_id = params["project_id"]

    url = f"https://cloudresourcemanager.googleapis.com/v1/projects/{project_id}:getIamPolicy"

    request_body = {}
    json_result = make_api_request(url, "POST", request_body)
    print(f"setProjectLevelIamPolicy (GET) json_result: {json_result}")

    # Test to see if permissions exist
    if "bindings" in json_result:
        for item in json_result["bindings"]:
            if item["role"] == role:
                members = item["members"]
                for member in members:
                    if member == accountWithPrefix:
                        print("Permissions exist")
                        return

    # Take the existing bindings and we need to append the new permission
    # Otherwise we loose the existing permissions
    if "bindings" in json_result:
        bindings = json_result["bindings"]
    else:
        bindings = []

    new_permission = {"role": role, "members": [accountWithPrefix]}

    bindings.append(new_permission)

    # https://cloud.google.com/resource-manager/reference/rest/v1/projects/setIamPolicy
    url = f"https://cloudresourcemanager.googleapis.com/v1/projects/{project_id}:setIamPolicy"

    request_body = {"policy": {"bindings": bindings}}

    print(f"Permission bindings: {bindings}")

    json_result = make_api_request(url, "POST", request_body)
    print()
    print(f"json_result: {json_result}")
    print()
    print(f"Project Level IAM Permissions set for {accountWithPrefix} {role}")


def create_text_embedding_model(params):
    """Creates a text embedding model."""
    project_id = params["project_id"]
    bigquery_location = params["bigquery_location"]
    vertex_ai_connection_name = params["vertex_ai_connection_name"]
    dataset_id = params["dataset_id"]
    sql = f"""CREATE MODEL IF NOT EXISTS `{project_id}.{dataset_id}.google-textembedding`
    REMOTE WITH CONNECTION `{project_id}.{bigquery_location}.{vertex_ai_connection_name}`
    OPTIONS (endpoint = 'text-embedding-004');"""
    client = bigquery.Client()
    client.query_and_wait(sql)
    print(f"Text embedding model created: {project_id}.{dataset_id}.google-textembedding")


def deploy_text_embedding_model(params):
    vertex_ai_service_account_id = create_vertex_ai_connection(params)
    set_project_level_iam_policy(
        params,
        f"serviceAccount:{vertex_ai_service_account_id}",
        "roles/aiplatform.user",
    )
    print("Waiting 60 seconds for IAM permissions to propagate...")
    time.sleep(60)
    create_text_embedding_model(params)
