##----- 폴더 카테고리 -----##
- data : 학습 시 사용한 데이터류 저장 / 현 데모코드에선 딱히 필요 X
- log : 데모 프로그램 실행시 로그 저장 폴더
- model : 통합프로그램 내에서 사용할 각 모델 저장
- utils : 연산식등 구동에 필요한 여러 유틸리티 포함
- web : 웹 통신등을 위한 프로그램 포함
- weights : 학습된 웨이트 저장
- frontend : 웹페이지를 위한 html파일 등 포함


##----- 환경 설치 및 프로그램 구동 순서 -----##
1. 가상환경 설치 -> 아나콘다 설치되어 있음을 가정
    - conda create -n 가상환경명 --file cam_test_env.txt

2. mjpg-streamer 설치 ( git 설치 되어있음을 가정 -> 위 가상환경에는 설치되어있음.)
    - sh ./mjpg_streamer_install.sh
    - #출처 : https://github.com/jacksonliam/mjpg-streamer

3. 카메라 스트리밍 서버 on
    - (가상환경 내에서) cam_test_server.py 실행


4. 카메라 데이터 가져오기
    - dlib 설치nvcc
    - (가상환경 내에서) cam_test_client.py 실행


##----- opencv err -----##
cv2.error: OpenCV(3.4.2) /tmp/build/80754af9/opencv-suite_1535558553474/work/modules/highgui/src/window.cpp:632: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Carbon support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'
위처럼, opencv는 정상적으로 improt 되는 것 처럼 나오지만 실제로 사용할 때 opencv 재빌드가 필요하다는 에러가 나올때 아래 폴더의 실행파일을 실행한다.
$: sh ./cam/opencv_err_fix.sh
# 출처 https://light-tree.tistory.com/150


##----- 아나콘다를 사용한 가상환경 패키지 리스트 추출 -----##
- 현재 가상환경에 설치된 패키지 목록 파일로 추출하는 방법
- conda list -n 가상환경명 --explicit>추출파일명
- ex) conda list -n MyEnv --explicit>env_pkg_list.txt
- 위 명령어를 이용하면 현재 가상환경을 복사할 수 있고,
- 패키지 리스트를 복사한 텍스트파일을 이용해 동일한 가상환경을 생성할 수 있다.


##----- dlib 설정 -----##
 - pip install dlib 로 설치하게 될 경우 gpu를 사용하지 못하는 버그가 발생
 - install_dlib.sh 파일을 적절한 위치에 이동시킨 후, sh 명령어를 이용해 설치.
 - 적절한 위치로 이동하는 이유는 해당 sh 파일 내에서 git clone을 사용해 새 디렉토리를 만들기 때문인데
 - 그냥 현재 디렉토리 위치에 생성해도 되지만 가능하면 따로 git clone 폴더를 관리하는 폴더를 생성하는게 깔끔하기 때문.


 ##----- cuda와 cudnn 등 설치하기 -----##
  - 우선 설치되었는지 확인하기
  - CUDA 설치 확인 
  - nvcc --version
  - sudo apt-get install nvidia-cuda-toolkit
  - --> 미설치시 설치가능한 명령어를 알려준다. 설치하자.
  - cudnn 설치확인
  - cat /usr/include/x86_64-linux-gnu/cudnn_v*.h | grep CUDNN_MAJOR -A 2
  - 

##----- 로컬파일 서버로 옮기기 -----##
 - scp -r "보낼 폴더" abr@155.230.14.96:~/union_workspace -P 8222


##----- 도커 gpu 못잡을 때 -----##
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker


##----- 도커 관련 -----##
https://hub.docker.com/repositories

