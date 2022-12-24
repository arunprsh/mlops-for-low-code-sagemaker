from sagemaker.workflow.parameters import ParameterInteger, ParameterString, ParameterFloat
from sagemaker.dataset_definition.inputs import AthenaDatasetDefinition, DatasetDefinition
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.steps import ProcessingStep, TransformStep
from sagemaker import MetricsSource, ModelMetrics, ModelPackage
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.automl_step import AutoMLStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.functions import Join
from sagemaker.transformer import Transformer
from sagemaker.pipeline import PipelineModel
from botocore.exceptions import ClientError
from sagemaker import AutoML, AutoMLInput
from sagemaker.model import Model
import sagemaker.session
import sagemaker
import logging
import boto3
import json
import time
import yaml
import os


logger = logging.getLogger('sagemaker')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def get_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket)


def get_pipeline_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket)


def resolve_ecr_uri_from_image_versions(sagemaker_session, image_versions, image_name):
    for image_version in image_versions:
        if image_version['ImageVersionStatus'] == 'CREATED':
            image_arn = image_version["ImageVersionArn"]
            version = image_version["Version"]
            logger.info(f"Identified the latest image version: {image_arn}")
            response = sagemaker_session.sagemaker_client.describe_image_version(
                ImageName=image_name,
                Version=version
            )
            return response['ContainerImage']
    return None


def resolve_ecr_uri(sagemaker_session, image_arn):
    image_name = image_arn.partition("image/")[2]
    try:
        next_token=''
        while True:
            response = sagemaker_session.sagemaker_client.list_image_versions(
                ImageName=image_name,
                MaxResults=100,
                SortBy='VERSION',
                SortOrder='DESCENDING',
                NextToken=next_token
            )
            ecr_uri = resolve_ecr_uri_from_image_versions(sagemaker_session, response['ImageVersions'], image_name)
            if "NextToken" in response:
                next_token = response["NextToken"]

            if ecr_uri is not None:
                return ecr_uri
        error_message = (
            f"No image version found for image name: {image_name}"
            )
        logger.error(error_message)
        raise Exception(error_message)

    except (ClientError, sagemaker_session.sagemaker_client.exceptions.ResourceNotFound) as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)
        
        
def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="LowCodePipelineModels",
    pipeline_name="LowCodePipeline",
    base_job_prefix="LowCode",
    project_id="SageMakerProjectId"):
    sagemaker_session = get_session(region, default_bucket)
    
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    pipeline_session = get_pipeline_session(region, default_bucket)
    s3_client = boto3.client('s3')
    current_timestamp = time.strftime('%d-%H-%M-%S', time.gmtime())
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    
    with open('./pipelines/lowcode/config/config.yml', 'r') as file_:
        config = yaml.safe_load(file_)
        
    instance_count = config['processing_step']['instance_count']
    instance_type = config['processing_step']['instance_type']
    container_uri = config['processing_step']['container_uri']
    bucket = config['processing_step']['bucket']
    metadata_folder = config['processing_step']['metadata_folder']

    dw_processing_instance_count = ParameterInteger(name='DWProcessingInstanceCount', default_value=instance_count)
    dw_processing_instance_type = ParameterString(name='DWProcessingInstanceType', default_value=instance_type)
    dw_container_uri = ParameterString(name='DWContainerURI', default_value=container_uri)
    bucket = ParameterString(name='Bucket', default_value=bucket)
    metadata_folder = ParameterString(name='MetadataFolder', default_value=metadata_folder)

    instance_type = config['model_creation_step']['instance_type']
    target_name = config['automl_step']['target_name']

    model_creation_instance_type = ParameterString(name='ModelCreationInstanceType', default_value=instance_type)
    target_name = ParameterString(name='TargetColumnName', default_value=target_name)

    instance_type = config['batch_transform_step']['instance_type']
    instance_count = config['batch_transform_step']['instance_count']

    batch_transform_instance_count = ParameterInteger(name='BatchTransformInstanceCount', default_value=instance_count)
    batch_transform_instance_type = ParameterString(name='BatchTransformInstanceType', default_value=instance_type)

    instance_type = config['evaluation_step']['instance_type']
    instance_count = config['evaluation_step']['instance_count']

    evaluation_processing_instance_count = ParameterInteger(name='EvaluationProcessingInstanceCount', default_value=instance_count)
    evaluation_processing_instance_type = ParameterString(name='EvaluationProcessingInstanceType', default_value=instance_type)

    status = config['register_step']['model_approval_status']
    group_name = config['register_step']['model_package_group_name']
    threshold = config['register_step']['model_registration_metric_threshold']

    model_approval_status = ParameterString(name='ModelApprovalStatus', default_value=status)
    model_package_group_name = ParameterString(name='ModelPackageName', default_value=group_name)
    model_registration_metric_threshold = ParameterFloat(name='ModelRegistrationMetricThreshold', default_value=threshold)
    
    input_folder = config['processing_step']['input_folder']
    output_folder = config['processing_step']['output_folder']
    input_name = config['processing_step']['input_name']
    node_id = config['processing_step']['node_id']
    output_name = f'{node_id}.default'
    input_path = f's3://{bucket.default_value}/{input_folder}'
    output_path = f's3://{bucket.default_value}/{output_folder}'
    flow_file_name = config['processing_step']['flow_file_name']

    ebs_volume_size = config['processing_step']['ebs_volume_size']
    output_content_type = config['processing_step']['output_content_type']
    is_refit = config['processing_step']['refit']
    
    data_sources = []
    processing_input = ProcessingInput(source=f'{input_path}/{input_name}', 
                                       destination=f'/opt/ml/processing/{input_name}', 
                                       input_name=input_name, 
                                       s3_data_type='S3Prefix', 
                                       s3_input_mode='File', 
                                       s3_data_distribution_type='FullyReplicated')
    data_sources.append(processing_input)
    
    processing_job_output = ProcessingOutput(source='/opt/ml/processing/output', 
                                         destination=f'{output_path}/{current_timestamp}',
                                         output_name=output_name,
                                         s3_upload_mode='EndOfJob')
    
    s3_client.upload_file(f'./pipelines/lowcode/config/{flow_file_name}',
                      bucket.default_value, 
                      f'{metadata_folder.default_value}/{current_timestamp}-{flow_file_name}')
    flow_S3_uri = f's3://{bucket.default_value}/{metadata_folder.default_value}/{current_timestamp}-{flow_file_name}'
    flow_input = ProcessingInput(source=flow_S3_uri, 
                             destination='/opt/ml/processing/flow', 
                             input_name='flow', 
                             s3_data_type='S3Prefix', 
                             s3_input_mode='File', 
                             s3_data_distribution_type='FullyReplicated')
    job_name = f'Data-Wrangler-Processing-job-{current_timestamp}'
    refit_trained_params = {'refit': is_refit, 
                        'output_flow': f'{current_timestamp}-refitted-{flow_file_name}'}
    processor = Processor(base_job_name=job_name,
                      role=role, 
                      image_uri=dw_container_uri, 
                      instance_count=dw_processing_instance_count, 
                      instance_type=dw_processing_instance_type, 
                      volume_size_in_gb=ebs_volume_size,  
                      sagemaker_session=pipeline_session)
    
    data_wrangler_step = ProcessingStep(name='DataWranglerProcessingStep', 
                                    processor=processor, 
                                    inputs=[flow_input] + data_sources, 
                                    outputs=[processing_job_output], 
                                    job_arguments=[f"--refit-trained-params '{json.dumps(refit_trained_params)}'"])
    input_content_type = config['automl_step']['input_content_type']
    auto_ml = AutoML(role=role, 
                 target_attribute_name=target_name, 
                 sagemaker_session=pipeline_session, 
                 mode='ENSEMBLING')
    s3_input = Join(on='/', 
                values=[data_wrangler_step.properties.ProcessingOutputConfig.Outputs[output_name].S3Output.S3Uri,
                        data_wrangler_step.properties.ProcessingJobName, 
                        f'{output_name.replace(".", "/")}'])
    train_args = auto_ml.fit(inputs=AutoMLInput(inputs=s3_input, 
                                            content_type=input_content_type, 
                                            target_attribute_name=target_name))
    automl_step = AutoMLStep(name='AutoMLStep', 
                         step_args=train_args)
    
    
    best_automl_model = automl_step.get_best_auto_ml_model(role, 
                                                       sagemaker_session=pipeline_session)
    
    
    
    best_inference_container = {
    'Image': best_automl_model.image_uri,
    'ModelDataUrl': best_automl_model.model_data,
    'Environment': best_automl_model.env}
    
    
    flow_tar_name = config['model_creation_step']['flow_tar_name']
    s3_client.upload_file(f'./pipelines/lowcode/config/{flow_tar_name}', 
                      bucket.default_value, 
                      f'{metadata_folder.default_value}/{current_timestamp}-{flow_tar_name}')
    inference_flow_uri = f's3://{bucket.default_value}/{metadata_folder.default_value}/{current_timestamp}-{flow_tar_name}'
    algo_container_uri = best_automl_model.image_uri
    algo_model_uri = best_automl_model.model_data
    pipeline_models = []
    data_wrangler_model_name = f"DataWranglerInferencePipelineFlowModel-{current_timestamp}"
    data_wrangler_model = Model(image_uri=dw_container_uri, 
                            model_data=inference_flow_uri, 
                            role=role, 
                            name=data_wrangler_model_name, 
                            sagemaker_session=pipeline_session, 
                            env={"INFERENCE_TARGET_COLUMN_NAME": target_name})
    pipeline_models.append(data_wrangler_model)
    algo_model_name = f"DataWranglerInferencePipelineAlgoModel-{current_timestamp}"
    algo_environment = best_inference_container["Environment"]

    algo_model = Model(image_uri=algo_container_uri, 
                   model_data=algo_model_uri, 
                   role=role, 
                   name=algo_model_name, 
                   sagemaker_session=pipeline_session, 
                   env=algo_environment)
    pipeline_models.append(algo_model)
    inference_pipeline_model_name = f"DataWranglerInferencePipelineModel-{current_timestamp}"
    inference_pipeline_model = PipelineModel(models=pipeline_models, 
                                         role=role, 
                                         name=inference_pipeline_model_name, 
                                         sagemaker_session=pipeline_session)
    step_args_create_model = inference_pipeline_model.create(instance_type=model_creation_instance_type)
    step_create_model = ModelStep(name='InferencePipeline', step_args=step_args_create_model)
    
    holdout_file_name = config['batch_transform_step']['holdout_file_name']
    s3_client.upload_file(f'./pipelines/lowcode/data/{holdout_file_name}', 
                      bucket.default_value, 
                      f'{metadata_folder.default_value}/{holdout_file_name}')
    holdout_s3_path = f's3://{bucket.default_value}/{metadata_folder.default_value}/{holdout_file_name}'

    true_labels_file_name = config['batch_transform_step']['true_labels_file_name']
    s3_client.upload_file(f'./pipelines/lowcode/data/{true_labels_file_name}', 
                      bucket.default_value, 
                      f'{metadata_folder.default_value}/{true_labels_file_name}')
    true_labels_s3_path = f's3://{bucket.default_value}/{metadata_folder.default_value}/{true_labels_file_name}'
    
    transformer = Transformer(model_name=step_create_model.properties.ModelName, 
                          instance_count=batch_transform_instance_count, 
                          instance_type=batch_transform_instance_type, 
                          output_path=Join(on='/', values=['s3:/', bucket, metadata_folder, 'transform']), 
                          sagemaker_session=pipeline_session)
    step_batch_transform = TransformStep(name='BatchTransformStep', 
                                     step_args=transformer.transform(data=holdout_s3_path, 
                                                                     content_type='text/csv'))
    evaluation_report = PropertyFile(name='evaluation', 
                                 output_name='evaluation_metrics', 
                                 path='evaluation_metrics.json')
    sklearn_processor = SKLearnProcessor(role=role, 
                                     framework_version='1.0-1', 
                                     instance_count=evaluation_processing_instance_count, 
                                     instance_type=evaluation_processing_instance_type, 
                                     sagemaker_session=pipeline_session)
    step_args_sklearn_processor = sklearn_processor.run(
    inputs=[
        ProcessingInput(
            source=step_batch_transform.properties.TransformOutput.S3OutputPath,
            destination='/opt/ml/processing/input/predictions',
        ),
        ProcessingInput(source=true_labels_s3_path, destination='/opt/ml/processing/input/true_labels'),
    ],
    outputs=[
        ProcessingOutput(
            output_name='evaluation_metrics',
            source="/opt/ml/processing/evaluation",
            destination=Join(on='/', values=['s3:/', bucket, metadata_folder, 'evaluation']),
        ),
    ],
    code=f'./pipelines/lowcode/src/evaluation.py')
    step_evaluation = ProcessingStep(name='ModelEvaluationStep', 
                                 step_args=step_args_sklearn_processor, 
                                 property_files=[evaluation_report])
    model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=automl_step.properties.BestCandidateProperties.ModelInsightsJsonReportPath,
        content_type='application/json'),
    explainability=MetricsSource(
        s3_uri=automl_step.properties.BestCandidateProperties.ExplainabilityJsonReportPath,
        content_type='application/json'))
    step_args_register_model = inference_pipeline_model.register(content_types=['text/csv'], 
                                                      response_types=['text/csv'], 
                                                      inference_instances=[batch_transform_instance_type], 
                                                      transform_instances=[batch_transform_instance_type], 
                                                      model_package_group_name=model_package_group_name, 
                                                      approval_status=model_approval_status, 
                                                      model_metrics=model_metrics)
    step_register_model = ModelStep(name='MeetsThreshold', 
                                step_args=step_args_register_model)
    step_conditional_registration = ConditionStep(name='ConditionalStep', 
                                              conditions=[ConditionGreaterThanOrEqualTo(
                                                            left=JsonGet(
                                                                step_name=step_evaluation.name,
                                                                property_file=evaluation_report,
                                                                json_path='classification_metrics.weighted_f1.value',
                                                            ),
                                                            right=model_registration_metric_threshold,
                                              )],
                                              if_steps=[step_register_model], 
                                              else_steps=[])
    pipeline_name = f'LowCodePipeline'
    pipeline_steps = [data_wrangler_step, automl_step, step_create_model, step_batch_transform, step_evaluation, step_conditional_registration]
    pipeline = Pipeline(name=pipeline_name, 
                    parameters=[dw_processing_instance_count, 
                                dw_processing_instance_type, 
                                dw_container_uri,
                                bucket, 
                                metadata_folder,
                                model_creation_instance_type,
                                target_name,
                                batch_transform_instance_count,
                                batch_transform_instance_type,
                                evaluation_processing_instance_count,
                                evaluation_processing_instance_type,
                                model_approval_status,
                                model_registration_metric_threshold, 
                                model_package_group_name,], 
                    steps=pipeline_steps, 
                    sagemaker_session=pipeline_session)
    return pipeline

    
    
    
    
    
    
    
    
    





    
   