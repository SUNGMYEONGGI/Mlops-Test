### Challenge9. MLOps

<h2>💁🏻‍♂️ 목표</h2>

MLOps는 DevOps 과정에 머신러닝을 트레이닝/테스트/모델생성/모델 배포 과정을 통합합니다. MLaaS 서비스를 활용해 어떻게 모델을 웹서비스에 함께 배포까지 가능한지 확인합니다.

<h2>📚 TO-DO</h2>

- [x] MLOps 이해
- [x] azureml examples 리포지토리 튜토리얼 노트북으로 기초 MLOps 과정 확인
  - [x] [https://github.com/CloudBreadPaPa/azureml-examples/](https://github.com/CloudBreadPaPa/azureml-examples/) - clone
  - [x] tutorials/get-started-notebooks/train-model.ipynb 노트북 실행
    - [ ] 전체 MLaaS 동작 과정을 이해 
  - [x] tutorials/get-started-notebooks/deploy-model.ipynb 노트북 실행
    - [x] 배포 과정을 수행한다. quota 이슈가 있을 수 있으니 주의
    - [ ] Endpoint instance 생성 주의
  - [ ] tutorials/get-started-notebooks/pipeline.ipynb 노트북 실행
    - [ ] 파이프라인 빌드 과정을 다룬다. 파이프라인을 publish는 안함

<h2>💡 힌트</h2>

**1️⃣ MLOps V2 리포지토리는 지속 업데이트됩니다. - [Azure MLOps (v2) solution accelerators](https://github.com/Azure/mlops-v2)**

**2️⃣ AzureML SDK는 V1과 V2가 있으며, 도전과제는 V2를 사용**

**3️⃣ AWS의 SageMaker나 Google Cloud의 Vertex AI도 AzureML과 거의 유사합니다.**
  - [MLOps example using Amazon SageMaker Pipeline and GitHub Actions](https://github.com/aws-samples/mlops-sagemaker-github-actions)
  - [An end-to-end example of MLOps on Google Cloud using TensorFlow, TFX, and Vertex AI](https://github.com/GoogleCloudPlatform/mlops-with-vertex-ai)

<h2>🧑🏻‍💻 개발단계</h2>

1️⃣ 공식 [repository](https://github.com/Azure/mlops-v2) 이동

2️⃣ template 복사를 수행

<img src="https://github.com/SUNGMYEONGGI/image/blob/main/%EA%B7%B8%EB%A6%BC1.png?raw=true" width="600" height="300">

3️⃣ Azure mlops-v2 / [git clone https://github.com/Azure/mlops-v2.git](https://github.com/Azure/mlops-v2) 

4️⃣ [가이드 문서 링크](https://github.com/Azure/mlops-v2/blob/main/documentation/deployguides/deployguide_gha.md
)

```Bash
# gh 설치
brew install gh
# WSL 우분투일 경우
https://github.com/cli/cli/blob/trunk/docs/install_linux.md#debian-ubuntu-linux-raspberry-pi-os-apt
# gh로 로그인 하려면
gh auth login
sudo apt-get install dos2unix
dos2unix sparse_checkout.sh
```

5️⃣ sparse 파일 수정 내역
```python
infrastructure_version=terraform   #options: terraform / bicep
project_type=classical   #options: classical / cv / nlp
mlops_version=aml-cli-v2   #options: aml-cli-v2 / python-sdk-v1 / python-sdk-v2 / rai-aml-cli-v2
orchestration=github-actions   #options: github-actions / azure-devops
git_folder_location='project root'   #replace with the local root
project_name=Mlops-Test   #replace with your project name
github_org_name=cloudbreadpapa-변경   #replace with your github org name
project_template_github_url=https://github.com/azure/mlops-project-template
```
```bash
bash sparse_checkout.sh
cd ~
mkdir project
```

6️⃣ github action을 위한 secret 생성 - AzureML이라 범주가 전체 구독 단위임
```bash
az ad sp create-for-rbac --name <service_principal_name> --role contributor --scopes /subscriptions/<subscription_id> --sdk-auth
```

7️⃣ 테라폼을 이용하므로 아래 네개의 값도 추가한다.
- AZURE_CREDENTIALS: 위의 JSON 전체 문자열
- ARM_CLIENT_ID
- ARM_CLIENT_SECRET
- ARM_SUBSCRIPTION_ID
- ARM_TENANT_ID

8️⃣ 파일 수정
`config-infra-prod.yml`, `config-infra-dev.yml` 두 파일존재
- 두 파일에서 `location: koreacentral`로 변경
- `namespace: mlopsv2`를 2개 글자 추가해 고유하게 생성
- namespace: `아무두글자` + opsv2
- main 브랜치와 dev 브랜치 둘 다 작업
- `.github/workflows/deploy-model-training-pipeline-classical.yml` 파일 내부의 내용 변경
  - `size: Standard_DS2_v2`  # DS2_v2로 줄임
  - `min_instances: 0`
  - `max_instances: 4`
  - `#cluster_tier: low_priority`  # 라인을 주석처리. 특수 quota가 필요
- `mlops/azureml/deploy/online/online-deployment.yml` 파일 내부의 내용 변경
  - `instance_type: Standard_DS3_v2` ➡️ `instance_type: Standard_DS2_v2` 

🔟 online - deployment를 수행
  - Github Actions CI/CD 작업
    
	<img src="https://github.com/SUNGMYEONGGI/image/blob/main/ci%20cd.png?raw=true" width="600" height="300">
  - [Azure Pipeline](https://ml.azure.com/experiments/id/9eaca022-f5d6-4908-87be-a06334f5cf42/runs/mighty_music_bkyccfdwf9?wsid=/subscriptions/3ebd25a6-6e5e-47b7-a80c-a3d971e2ca19/resourcegroups/rg-dlopsv2-0001dev/providers/Microsoft.MachineLearningServices/workspaces/mlw-dlopsv2-0001dev&tid=478b1b0f-75b7-4db5-9952-6c7a708d98a6#/?graphId=e2e53e3f-0d75-4af3-acd8-f09b8d9a0552&label=mighty_music_bkyccfdwf9&newGraphId=e2e53e3f-0d75-4af3-acd8-f09b8d9a0552&path=%2Fexperiments%2Fid%2F9eaca022-f5d6-4908-87be-a06334f5cf42%2Fruns%2Fmighty_music_bkyccfdwf9&runId=mighty_music_bkyccfdwf9) 확인
    
	<img src="https://github.com/SUNGMYEONGGI/image/blob/main/pipeline.png?raw=true" width="600" height="300">
</br>

## 💣 Issue

### Issue 1
<train 파이프라인을 돌리면 에러가 발생>

```The specified subscription has a total vCPU quota of 0```

✅ MSDN 구독 계정일 경우 쿼터 제한이슈. `cluster_tier: low_priority` 사용시 발생

### Issue 2
아래 오류가 발생한다면 `Standard_DS3_v2`가 문제다. 
```bash
VmSize":["Not enough quota available for Standard_DS3_v2 in SubscriptionId ***. Current usage/limit: 0/6. Additional needed: 8 Please see troubleshooting guide, available her
"errors":***"VmSize":["Not enough quota available for Standard_DS3_v2 in SubscriptionId ***. Current usage/limit: 0/6. Additional needed: 8 Please see troubleshooting guide, available here: https://aka.ms/oe-tsg#error-outofquota"]*
```
✅ 4개 * 2 해서 8개 cpu를 사용해 내 쿼터 가용 cpu 6개를 넘는다. VM Size를 `Standard_DS2_v2`으로 변경

### Issue 3
online - deployment 오류시
ERROR: (UserError) An endpoint with this name already exists. If you are trying to create a new endpoint, use a
azureml 스튜디오에서 endpoint를 삭제하고 다시 실행

✅ 잊지 말고, dev branch를 생성하고 dev로 deploy 한다.
