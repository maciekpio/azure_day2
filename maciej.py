from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment
from azure.ai.ml import command
from azure.ai.ml import Input, Output
from azure.ai.ml import load_component
from azure.ai.ml import dsl
import webbrowser
import argparse

import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help="env")
    args = parser.parse_args()
    subscription_id = "59a62e46-b799-4da2-8314-f56ef5acf82b"
    resource_group = "rg-azuremltraining"
    workspace = "dummy-workspace"
    ml_client = MLClient(
        DefaultAzureCredential(), subscription_id, resource_group, workspace
    )

    web_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

    credit_data = Data(
        path=web_path,
        type=AssetTypes.URI_FILE,
        name="maciej_creditcard_defaults_"+str(args.env),
        tags={"creator": "maciej"}
    )

    ml_client.data.create_or_update(credit_data)
    print(
        f"Dataset with name {credit_data.name} was registered to workspace, the dataset version is {credit_data.version}"
    )

    cpu_compute_target = "aml-cluster"

    dependencies_dir = "./dependencies"

    custom_env_name = "aml-scikit-learn"
    # Create an Environment based on the conda yaml specification above
    # As a base image, use "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    # Name it <your_name>_environment
    # Also tag it with a tag "creator" with your name as a value
    pipeline_job_env = Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        conda_file="dependencies/conda.yaml",
        name="maciej_environment_"+str(args.env),
        tags={"creator": "maciej"}
    )
    ml_client.environments.create_or_update(pipeline_job_env)

    data_prep_src_dir = "./components/data_prep"

    data_prep_component = command(
        name="maciej_data_prep_credit_defaults_"+str(args.env),
        display_name="Data preparation for training",
        description="reads a .xl input, split the input to train and test",
        inputs={
            "data": Input(type="uri_folder"),
            "test_train_ratio": Input(type="number"),
        },
        outputs=dict(
            train_data=Output(type="uri_folder", mode="rw_mount"),
            test_data=Output(type="uri_folder", mode="rw_mount"),
        ),
        # The source folder of the component
        code=data_prep_src_dir,
        command="""python data_prep.py \
                --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} \
                --train_data ${{outputs.train_data}} --test_data ${{outputs.test_data}} \
                """,
        environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
    )

    data_prep_component = ml_client.create_or_update(data_prep_component.component)
    print(
        f"Component {data_prep_component.name} with Version {data_prep_component.version} is registered"
    )

    train_src_dir = "./components/train"

    train_component = load_component(source=os.path.join(train_src_dir, "train.yml"))

    train_component = ml_client.create_or_update(train_component)
    print(
        f"Component {train_component.name} with Version {train_component.version} is registered"
    )

    @dsl.pipeline(
        compute=cpu_compute_target,
        description="E2E data_perp-train pipeline",
    )
    def credit_defaults_pipeline(
        pipeline_job_data_input,
        pipeline_job_test_train_ratio,
        pipeline_job_learning_rate,
        pipeline_job_registered_model_name,
    ):
        # using data_prep_function like a python call with its own inputs
        # Add the data prep and training component
        data_prep_job=data_prep_component(data=pipeline_job_data_input,test_train_ratio=pipeline_job_test_train_ratio)
        print(data_prep_job)
        train_job=train_component(train_data=data_prep_job.outputs.train_data,
                                test_data=data_prep_job.outputs.test_data,
                                learning_rate=pipeline_job_learning_rate,
                                registered_model_name=pipeline_job_registered_model_name)
        return {
            "pipeline_job_train_data": data_prep_job.outputs.train_data,
            "pipeline_job_test_data": data_prep_job.outputs.test_data,
        }

    registered_model_name = "maciej_credit_defaults_model_"+str(args.env)

    pipeline = credit_defaults_pipeline(
        pipeline_job_data_input=Input(type="uri_file", path=credit_data.path),
        pipeline_job_test_train_ratio=0.25,
        pipeline_job_learning_rate=0.05,
        pipeline_job_registered_model_name=registered_model_name,
    )

    #SUBMITTING THE JOBBBBB
    if str(args.env)!="prd":
        pipeline_job = ml_client.jobs.create_or_update(
            pipeline,
            experiment_name="e2e_registered_components_maciej_"+str(args.env),
        )
        # open the pipeline in web browser
        webbrowser.open(pipeline_job.studio_url)

if __name__ == "__main__":
    main()