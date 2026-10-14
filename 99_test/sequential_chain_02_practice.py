# ==========================================================================
# [실습] LCEL Sequential Chain - 해외 고객 문의(CS) 이메일 자동 처리
# --------------------------------------------------------------------------
# 시나리오:
#   온라인 쇼핑몰에 해외 고객의 문의 이메일이 접수되었습니다.
#   LLM 체인을 이용해 아래 6단계를 순차적으로 처리하세요.
#     1) 문의 이메일을 한국어로 번역
#     2) 번역된 문의를 한 문장으로 요약
#     3) 번역된 문의의 긴급도를 0~10점으로 평가
#     4) 원본 이메일에 사용된 언어 식별
#     5) 요약 내용에 대해 원본 언어로 정중한 답변 작성
#     6) 생성된 답변을 한국어로 번역 (담당자 검토용)
#
# ※ 아래 주석을 참고하여 각 위치에 코드를 직접 작성하세요.
# ==========================================================================


# --- 1. 라이브러리 임포트 ---
# os, dotenv(load_dotenv)를 임포트하세요.
# langchain_core.prompts 에서 PromptTemplate 을 임포트하세요.
# langchain_core.output_parsers 에서 StrOutputParser 를 임포트하세요.
# langchain_core.runnables 에서 RunnablePassthrough 를 임포트하세요.
# langchain_openai 에서 ChatOpenAI 를 임포트하세요.

import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


# --- 2. 환경 변수 로드 ---
# load_dotenv()로 .env 파일을 불러오세요.
# os.getenv()로 OPENAI_API_KEY 값을 가져와 변수에 저장하세요.
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- 3. LLM 모델 초기화 ---
# ChatOpenAI 객체를 생성하세요.
#   - model: "gpt-3.5-turbo" (또는 사용 가능한 모델)
#   - api_key: 위에서 불러온 키
#   - temperature: 0.7 (응답의 다양성 조절)

llm = ChatOpenAI(
    model="gpt-3.5-turbo"
    api_key=OPENAI_API_KEY
    temperature=0.7
)

# --- 4. 프롬프트 템플릿 정의 ---

# 프롬프트 1: 문의 이메일 번역
#   - PromptTemplate(input_variables=[...], template=...) 형태로 작성
#   - 입력 변수: inquiry
#   - 지시 내용: "다음 고객 문의 이메일을 한국어로 번역하세요."

prompt1 = PromptTemplate(
    input_variables=["inquiry"],
    template="다음 고객 문의 이메일을 한국어로 번역하세요.\n\n{inquiry}"
)

# 프롬프트 2: 번역된 문의 요약
#   - PromptTemplate.from_template() 사용
#   - 입력 변수: translation
#   - 지시 내용: "다음 고객 문의를 한 문장으로 요약하세요."

# 프롬프트 3: 긴급도 점수 평가
#   - 입력 변수: translation
#   - 지시 내용: "다음 고객 문의를 읽고 처리 긴급도를 0점(여유)부터 10점(매우 긴급) 사이로
#                평가하세요. 숫자만 대답하세요."

# 프롬프트 4: 원본 이메일 언어 식별
#   - 입력 변수: inquiry (번역본이 아닌 원본을 사용해야 함에 주의)
#   - 지시 내용: "다음 이메일에 사용된 언어가 무엇인가요? 언어 이름만 답하세요."

# 프롬프트 5: 고객 답변 작성 (원본 언어로)
#   - 입력 변수: language, summary
#   - 지시 내용: "다음 고객 문의 요약에 대해 고객센터 담당자로서 정중한 답변을 작성하세요.
#                답변 언어:{language} / 문의 요약:{summary}"

# 프롬프트 6: 답변을 한국어로 번역
#   - 입력 변수: reply1
#   - 지시 내용: "다음 답변을 한국어로 번역해주세요."


# --- 5. LCEL 체인 구성 요소 정의 ---
# 각 단계는 '프롬프트 | LLM | StrOutputParser()' 형태로 구성합니다.

# 단계 1: 번역 체인       -> translate_chain
# 단계 2: 요약 체인       -> summarize_chain
# 단계 3: 긴급도 평가 체인 -> urgency_chain
# 단계 4: 언어 식별 체인   -> language_chain
# 단계 5: 답변 생성 체인   -> reply1_chain
# 단계 6: 답변 번역 체인   -> reply2_chain


# --- 6. 전체 체인 결합 ---
# RunnablePassthrough.assign()을 파이프(|)로 연결하여 combined_chain 을 만드세요.
# 각 단계의 출력이 딕셔너리에 누적되어 다음 단계의 입력으로 전달됩니다.
#
# [1단계] translation 키 추가
#   - 입력: {'inquiry': ...}
#   - translate_chain 에 x["inquiry"] 전달
#   - 출력: {'inquiry', 'translation'}
#
# [2단계] summary, urgency_score, language 키를 한 번에 추가
#   - summary       : summarize_chain 에 x["translation"] 전달
#   - urgency_score : urgency_chain 에 x["translation"] 전달
#   - language      : language_chain 에 x["inquiry"] 전달 (원본 사용)
#   - 출력: {'inquiry', 'translation', 'summary', 'urgency_score', 'language'}
#
# [3단계] reply1 키 추가
#   - reply1_chain 에 x["language"], x["summary"] 전달
#
# [4단계] reply2 키 추가
#   - reply2_chain 에 x["reply1"] 전달


# --- 7. 입력 데이터 ---
# 아래 영문 문의 이메일을 inquiry_text 변수에 여러 줄 문자열(""" """)로 저장하세요.
#
#   Hello, I ordered a pair of running shoes (Order #A1029) two weeks ago,
#   but the package still hasn't arrived. The tracking page hasn't updated in 5 days.
#   I need them for a marathon next weekend.
#   Could you please check the delivery status or arrange a refund if it's lost?


# --- 8. 체인 실행 및 결과 출력 ---
# try-except 블록 안에서 실행하세요.
#   - combined_chain.invoke(input={'inquiry': inquiry_text}) 로 체인을 실행합니다.
#   - 결과 딕셔너리에서 .get()을 사용해 아래 값을 출력합니다. (없으면 "N/A")
#       translation, summary, urgency_score, language, reply1, reply2
#   - 예외 발생 시 f"Error: {e}" 형태로 오류 메시지를 출력합니다.
