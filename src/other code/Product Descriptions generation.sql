SELECT
  product_id,
  name,
  AI.GENERATE(
    ('Give a short, two sentence description of ', name),
    connection_id => 'us.gemini-bq-conn',
    endpoint => 'gemini-2.0-flash').result
FROM `esdemo-389207.retail_data.products`;

-- update products table column product_desc using product_desc table join on product_id
UPDATE
  `esdemo-389207`.`retail_data`.`products` AS products
SET
  product_desc = product_desc.result
FROM
  `esdemo-389207`.`retail_data`.`product_desc` AS product_desc
WHERE
  products.product_id = product_desc.product_id;

-- create remote text embeddings model
CREATE OR REPLACE MODEL `esdemo-389207.retail_data.google-textembedding`
REMOTE WITH CONNECTION `us.gemini-bq-conn`
OPTIONS (ENDPOINT = 'text-embedding-004');

-- create product description embeddings
CREATE OR REPLACE TABLE 
                `esdemo-389207.retail_data.products_description_embeddings` AS
                SELECT product_id, ml_generate_embedding_result as embedding
                FROM ML.GENERATE_EMBEDDING(
                    MODEL `esdemo-389207.retail_data.google-textembedding`,
                    (SELECT product_id, product_desc as content
                    FROM `esdemo-389207.retail_data.products`),
                    STRUCT(
                      TRUE AS flatten_json_output,
                      'SEMANTIC_SIMILARITY' as task_type,
                      768 AS output_dimensionality
                      ));

--User Query matched to Products Catalog using Embeddings

SELECT base.product_id,
       products.product_desc
    FROM
      VECTOR_SEARCH( TABLE `esdemo-389207.retail_data.products_description_embeddings`,
        'embedding',
        (
        SELECT
          text_embedding,
          content AS query
        FROM
          ML.GENERATE_TEXT_EMBEDDING( MODEL `esdemo-389207.retail_data.google-textembedding`,
            (
            SELECT 'timeless tropical accessories' AS content)) ),
        top_k => 3) 
    JOIN
      `esdemo-389207.retail_data.products` AS products
    ON
      base.product_id = products.product_id;


SELECT base.product_id,
       products.product_desc
    FROM
      VECTOR_SEARCH( TABLE `esdemo-389207.retail_data.products_description_embeddings`,
        'embedding',
        (
        SELECT
          text_embedding,
          content AS query
        FROM
          ML.GENERATE_TEXT_EMBEDDING( MODEL `esdemo-389207.retail_data.google-textembedding`,
            (
            SELECT 'Spring dressess for vacation' AS content)) ),
        top_k => 3) 
    JOIN
      `esdemo-389207.retail_data.products` AS products
    ON
      base.product_id = products.product_id;
