# 시연 서버 실행 (Mac)

녹음 → 인식 → 고치기 → **확인한 문장만** 읽기. 서버 하나와 웹 페이지 하나다.

## 1. 처음 한 번

```bash
cd ~/Desktop/sw_challenge/demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 실행

```bash
cd ~/Desktop/sw_challenge/demo
source .venv/bin/activate
python3 server.py
```

처음 실행하면 `openai/whisper-small`(약 1 GB)을 받는다. `open http://127.0.0.1:8000` 이 찍히면 브라우저에서 그 주소를 연다.
마이크 권한을 물으면 허용한다. 한국어 음성으로 읽으려면 macOS에 한국어 음성(예: Yuna)이 있어야 한다.

모델 없이 화면과 API만 확인하려면 `ASR_ENGINE=mock python3 server.py`.

## 3. 개인 어댑터 붙이기

Colab B1 실행이 끝나면 `results/<실행이름>/adapter/` 폴더를 여기로 복사한다.

```
demo/adapters/KJW_nall_s0/   ← adapter_config.json, adapter_model.safetensors, meta.json
demo/adapters/CYU_nall_s0/
```

서버를 다시 켜면 `loaded adapter KJW_nall_s0 (KJW)`처럼 찍힌다. 화면에서 사용자 KJW를 고르면 그 어댑터로, guest면 기본 모델로 인식한다.
`meta.json`의 `user_id`가 `null`이면(개발 세트에서 기준 모델을 못 이긴 어댑터) 서버가 읽지 않는다. 의도된 동작이다.

## 4. 시연 순서 (3분 안)

1. 데이터셋 샘플에서 KJW 문장 하나 선택 → **개인 어댑터 끄고** 인식 → 결과
2. 같은 문장을 **어댑터 켜고** 인식 → 결과 비교. 화면에 라벨(의도 기준 전사)이 같이 나온다
3. 틀린 글자를 직접 고친다 → "이 문장이 맞아요" → "읽어주기"
4. 확인 뒤 한 글자를 더 고치면 "읽어주기"가 잠기는 것을 보여준다 — 확인 안 된 문장은 소리로 나가지 않는다

**팀원이 구음장애 발화를 흉내 내서 성능을 보여주면 안 된다.** 샘플은 AI-Hub 데이터셋의 실제 음성(평가 세트, 학습에 안 쓴 문장)이다. 마이크 녹음은 흐름을 보여주는 용도로만 쓴다.

## 5. 확인

```bash
python3 test_contract.py http://127.0.0.1:8000 KJW
```
`0 failed`가 나와야 한다. 어댑터가 없으면 마지막 인자(KJW)를 뺀다.

## 알려진 한계

- 서버는 WAV 16 kHz 모노 16비트, 0.3~30초만 받는다. 웹 페이지가 알아서 변환한다.
- 재시작하면 인식·확인 기록이 사라진다(메모리 저장).
- 후보 여러 개와 신뢰도 점수는 v1에 없다. 화면에 "후보 0개 · 점수 없음"으로 나오는 게 정상이다.
- 실제 모델로는 아직 한 번도 안 돌려봤다. 이 코드는 모의 엔진으로만 검증했다(작업 환경에서 모델 다운로드가 막혀 있었다). 첫 실행에서 오류가 나면 메시지를 그대로 가져오면 된다.
