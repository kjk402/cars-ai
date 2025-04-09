# 베이스 이미지
FROM python:3.10-slim

# 시스템 패키지 업데이트 및 한글 폰트 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        fonts-nanum \
        fontconfig \
        wget && \
    fc-cache -fv && \
    rm -rf /root/.cache/matplotlib/* && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# start.sh 실행 권한 부여
RUN chmod +x ./start.sh

# 포트 오픈
EXPOSE 8001

# 실행 스크립트로 변경
CMD ["./start.sh"]

