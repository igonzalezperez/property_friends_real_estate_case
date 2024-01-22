# Property friends real estate case

Estimate property valuations

Property Friends Real Estate Case is a FastAPI-based application designed to receive inputs and use a trained machine learning model to return predictions based on those inputs. The application is containerized using Docker for ease of setup and reproducibility.

## Development Requirements

- Git (to clone the repository)
- Docker and Docker Compose
- Other requirements are handled within the container (e.g. python dependencies) so the user/host doesn't need them.

## Setup
1. Clone the repository using Git.
```
git clone https://github.com/igonzalezperez/property_friends_real_estate_case.git
```
2. Ensure Docker and Docker Compose are installed on your system.
3. Create a `.env` file in the project root directory using the values from `.env.example`.
4. Place your `train.csv` and `test.csv` files in the `/ml/data/raw/` directory.

## Running the app
### Data and model setup
1. Start the containers:
```
docker-compose up -d --build
```
2. Initialize postgres DB:
```
docker exec -it property_friends_real_estate_case-db-1 /bin/bash
```
This will open a CLI in the DB container, in there run:
```
bash ml/pipelines/db_init.sh
```
Then exit the container with:
```
exit
```
3. Run ML pipelines to load data, transform it and train predictive model.
```
docker exec -it property_friends_real_estate_case-app-1 /bin/bash
```
This will open a CLI in the DB container, in there run:
```
make ml-pipeline
```
When the pipeline finishes, the model will be stored and available for the app endpoints.
### App usage
Go to `localhost:8080/docs`, this will show the docs of the API using the swagger UI, in there you can test all the endpoints.
1. Upon starting the app, a valid API key will be created and stored at `app/valid_keys/api_keys.json`, there you can copy the API key and paste when prompted for it in the Authorize button of the UI.
2. Click on the endpoint entries, each one has a "Try it out" button, press it and then press the "Execute" button to test the endpoint with a valid payload. These are the available endpoints:
- /predict: POST a valid payload to receive a prediction.
- /health: GET the health status of the application.
- /predict-logs: GET recent calls to the predict endpoint.
- /model-runs: GET metadata of model training instances.

### M.L Model Environment

```sh
MODEL_PATH=./ml/model/
BASE_MODEL_NAME=model.joblib
```

### Update `/predict`

To update your machine learning model, add your `load` and `method` [change here](app/api/routes/predictor.py#L19) at `predictor.py`

## Installation

```sh
python -m venv venv
source venv/bin/activate
make install
```

## Runnning Localhost

`make run`

## Deploy app

`make deploy`

## Running Tests

`make test`

## Access Swagger Documentation

> <http://localhost:8080/docs>

## Access Redocs Documentation

> <http://localhost:8080/redoc>

## Project structure

Files related to application are in the `app` or `tests` directories.
Application parts are:

    app
    |
    | # Fast-API stuff
    ├── api                 - web related stuff.
    │   └── routes          - web routes.
    ├── core                - application configuration, startup events, logging.
    ├── models              - pydantic models for this application.
    ├── services            - logic that is not just crud related.
    ├── main-aws-lambda.py  - [Optional] FastAPI application for AWS Lambda creation and configuration.
    └── main.py             - FastAPI application creation and configuration.
    |
    | # ML stuff
    ├── data             - where you persist data locally
    │   ├── interim      - intermediate data that has been transformed.
    │   ├── processed    - the final, canonical data sets for modeling.
    │   └── raw          - the original, immutable data dump.
    │
    ├── notebooks        - Jupyter notebooks. Naming convention is a number (for ordering),
    |
    ├── ml               - modelling source code for use in this project.
    │   ├── __init__.py  - makes ml a Python module
    │   ├── pipeline.py  - scripts to orchestrate the whole pipeline
    │   │
    │   ├── data         - scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features     - scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   └── model        - scripts to train models and make predictions
    │       ├── predict_model.py
    │       └── train_model.py
    │
    └── tests            - pytest

## GCP

Deploying inference service to Cloud Run

### Authenticate

1. Install `gcloud` cli
2. `gcloud auth login`
3. `gcloud config set project <PROJECT_ID>`

### Enable APIs

1. Cloud Run API
2. Cloud Build API
3. IAM API

### Deploy to Cloud Run

1. Run `gcp-deploy.sh`

### Clean up

1. Delete Cloud Run
2. Delete Docker image in GCR

## AWS

Deploying inference service to AWS Lambda

### Authenticate

1. Install `awscli` and `sam-cli`
2. `aws configure`

### Deploy to Lambda

1. Run `sam build`
2. Run `sam deploy --guiChange this portion for other types of models

## Add the correct type hinting when completed

`aws cloudformation delete-stack --stack-name <STACK_NAME_ON_CREATION>`

Made by <https://github.com/arthurhenrique/cookiecutter-fastapi/graphs/contributors> with ❤️
