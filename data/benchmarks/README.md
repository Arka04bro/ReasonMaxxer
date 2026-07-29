Place optional local benchmark JSON files here for:

- `aime24.json`
- `amc23.json`
- `minerva_math.json`
- `olympiadbench.json`

Each file should be either a JSON list of records or a JSON object with a `records` field.
Each record should contain:

- `problem_id`
- `problem_text`
- `ground_truth`
- optional `category`
