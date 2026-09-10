import sys
import argparse
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost

def get_pipeline(region="eu-west-3", role_arn=None, bucket_name=None):
    sagemaker_session = sagemaker.Session()
    
    if not role_arn:
        role_arn = sagemaker.get_execution_role()
    if not bucket_name:
        bucket_name = sagemaker_session.default_bucket()

    # 1. Processing Step (Nettoyage & Feature Engineering Titanic via Scikit-Learn)
    sklearn_processor = SKLearnProcessor(
        framework_version='1.0-1',
        role=role_arn,
        instance_type='ml.m5.xlarge',
        instance_count=1,
        sagemaker_session=sagemaker_session
    )
    step_process = ProcessingStep(
        name="PreprocessDataStep",
        processor=sklearn_processor,
        code="scripts/preprocess.py",
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/output/test")
        ]
    )

    # 2. Training Step (Entraînement XGBoost)
    xgb_estimator = XGBoost(
        entry_point="scripts/train.py",
        framework_version="1.5-1",
        role=role_arn,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        base_job_name="xgboost-titanic",
        sagemaker_session=sagemaker_session,
        output_path=f"s3://{bucket_name}/artifacts",
        hyperparameters={
            "n_estimators": "100",
            "max_depth": "4",
            "learning_rate": "0.1"
        }
    )

    step_train = TrainingStep(
        name="TrainModelStep",
        estimator=xgb_estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv"
            ),
            "test": sagemaker.inputs.TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )

    # 3. Model Registry Step
    step_register = ModelStep(
        name="RegisterModelStep",
        step_args=xgb_estimator.register(
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.m5.xlarge"],
            transform_instances=["ml.m5.xlarge"],
            model_package_group_name="NativeAWSMLOpsGroup",
            approval_status="PendingManualApproval"
        )
    )

    # Assemblage
    pipeline = Pipeline(
        name="NativeSageMakerPipeline",
        steps=[step_process, step_train, step_register],
        sagemaker_session=sagemaker_session
    )
    return pipeline

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", type=str, required=True)
    parser.add_argument("--bucket-name", type=str, required=True)
    parser.add_argument("--region", type=str, default="eu-west-3")
    args = parser.parse_args()

    pipeline = get_pipeline(region=args.region, role_arn=args.role_arn, bucket_name=args.bucket_name)
    pipeline.upsert(role_arn=args.role_arn)
    print(f"Pipeline SageMaker 'NativeSageMakerPipeline' (XGBoost Titanic) mis à jour avec succès sur {args.region}.")