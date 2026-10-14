import os                                                   # 운영체제 환경변수를 다루는 기본 모듈
from dotenv import load_dotenv                              # .env 파일을 읽어 주는 함수
from langchain_core.prompts import PromptTemplate           # 프롬프트 '틀'을 만드는 클래스
from langchain_core.output_parsers import StrOutputParser   # 모델 응답에서 글자만 뽑아 주는 파서
from langchain_core.runnables import RunnablePassthrough    # 값을 통과시키며 새 키를 붙이는 부품
from langchain_openai import ChatOpenAI                     # OpenAI 채팅 모델 클래스

# 1) .env 파일의 내용을 환경변수로 등록합니다.
#    반드시 os.getenv() 보다 먼저 실행해야 합니다.
load_dotenv()

# 2) 등록된 환경변수 중 OPENAI_API_KEY 값을 꺼냅니다.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 3) 사용할 LLM 모델을 준비합니다.
#    - model       : 모델 이름
#    - api_key     : 위에서 꺼낸 비밀 키
#    - temperature : 응답 다양성 (0 = 안정적, 1 = 창의적)
openai_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0.7
)

# ===== 프롬프트 1: 리뷰 번역 =====
# 원본 리뷰(review)를 받아 한글로 번역합니다.
# input_variables 로 사용할 변수 이름을 직접 알려 주는 방식입니다.
prompt1 = PromptTemplate(
    input_variables=['review'],
    template="다음 숙박 시설 리뷰를 한글로 번역하세요.\n\n{review}"
)

# ===== 프롬프트 2: 번역된 리뷰 요약 =====
# 번역본(translation)을 받아 한 문장으로 요약합니다.
# from_template 은 {translation} 을 자동으로 변수로 인식합니다.
prompt2 = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰를 한 문장으로 요약하세요.\n\n{translation}"
)

# ===== 프롬프트 3: 감성 점수 평가 =====
# 번역본(translation)을 읽고 0~10점 사이의 숫자만 답하게 합니다.
prompt3 = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰를 읽고 0점부터 10점 사이에서 부정/긍정 점수를 매기세요. "
    "숫자만 답하세요.\n\n{translation}"
)


# ===== 프롬프트 4: 원본 리뷰 언어 식별 =====
# 주의! 번역본이 아니라 원본(review)을 넣어야 원래 언어를 알 수 있습니다.
prompt4 = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰에 사용된 언어가 무엇인가요? 언어 이름만 답하세요.\n\n{review}"
)

# ===== 프롬프트 5: 공손한 답변 생성 (원본 언어 사용) =====
# 변수 두 개를 사용합니다.
#   - language : 프롬프트 4가 알아낸 언어
#   - summary  : 프롬프트 2가 만든 요약
prompt5 = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰 요약에 대해 공손한 답변을 작성하세요.\n"
    "답변 언어:{language}\n"
    "리뷰 요약:{summary}"
)

# ===== 프롬프트 6: 생성된 답변을 한국어로 번역 =====
# 프롬프트 5가 만든 첫 번째 답변(reply1)을 받아 한국어로 번역합니다.
prompt6 = PromptTemplate.from_template(
    "다음 생성된 답변을 한국어로 번역해주세요.\n리뷰 번역 {reply1}"
)

# ===== LCEL 체인 구성 =====
# 공식: 프롬프트 | LLM | 출력 파서
#   ① 프롬프트가 빈칸을 채워 완성된 지시문을 만들고
#   ② LLM 이 그 지시문으로 답을 생성하고
#   ③ 파서가 답에서 글자만 뽑아 문자열로 돌려줍니다.

# 단계 1: 리뷰 번역 체인
translate_chain_component = prompt1 | openai_llm | StrOutputParser()

# 단계 2: 번역된 리뷰 요약 체인
summarize_chain_component =  prompt2 | openai_llm | StrOutputParser()

# 단계 3: 감성 점수 평가 체인
sentiment_score_chain_component =   prompt3 | openai_llm | StrOutputParser()

# 단계 4: 언어 식별 체인
language_chain_component =  prompt4 | openai_llm | StrOutputParser()

# 단계 5: 첫 번째 답변(원본 언어) 생성 체인
reply1_chain_component =  prompt5 | openai_llm | StrOutputParser()

# 단계 6: 두 번째 답변(한국어 번역) 생성 체인
reply2_chain_component =  prompt6 | openai_llm | StrOutputParser()


# print(reply2_chain_component.invoke({"review": "The room was clean."}))

# RunnablePassthrough.assign 으로 각 단계의 출력을 다음 단계 입력으로 넘기고,
# 중간 결과를 하나의 딕셔너리에 차곡차곡 누적합니다.
combined_lcel_chain = (

    # ① 첫 번째 assign
    # 입력: {'review': '원본 리뷰 텍스트'}
    # 출력: {'review': '...', 'translation': '번역된 텍스트'}
    RunnablePassthrough.assign(
        translation=lambda x: translate_chain_component.invoke({"review": x["review"]})
    )

    # ② 두 번째 assign — 서로 의존하지 않는 세 작업을 함께 처리
    # 출력: 위 값 + {'summary': ..., 'sentiment_score': ..., 'language': ...}
    | RunnablePassthrough.assign(
        summary=lambda x: summarize_chain_component.invoke({"translation": x["translation"]}),
        sentiment_score=lambda x: sentiment_score_chain_component.invoke({"translation": x["translation"]}),
        language=lambda x: language_chain_component.invoke({"review": x["review"]})   # 언어 감지만 원본 사용
    )

    # ③ 세 번째 assign
    # 출력: 위 값 + {'reply1': '원본 언어로 쓴 첫 번째 답변'}
    | RunnablePassthrough.assign(
        reply1=lambda x: reply1_chain_component.invoke({
            "language": x["language"],
            "summary": x["summary"]
        })
    )

    # ④ 네 번째 assign
    # 출력: 위 값 + {'reply2': '한국어로 번역된 두 번째 답변'}
    | RunnablePassthrough.assign(
        reply2=lambda x: reply2_chain_component.invoke({"reply1": x["reply1"]})
    )
)

# ===== 숙박 시설 리뷰 입력 =====
review_text = """
The hotel was clean and the staff were very helpful.
The location was convenient, close to many attractions.
However, the room was a bit small and the breakfast options were limited.
Overall, a decent stay but there is room for improvement.
"""

# ===== 체인 실행 및 결과 출력 =====
try:
    # 초기 입력을 딕셔너리로 전달해 전체 체인을 실행합니다.
    result = combined_lcel_chain.invoke(input={'review': review_text})

    # 결과 딕셔너리에서 키별로 값을 꺼내 출력합니다.
    print(f'translation 결과: {result.get("translation", "N/A")} \n')
    print(f'summary 결과: {result.get("summary", "N/A")} \n')
    print(f'sentiment_score 결과: {result.get("sentiment_score", "N/A")} \n')
    print(f'language 결과: {result.get("language", "N/A")} \n')
    print(f'reply1 결과: {result.get("reply1", "N/A")} \n')
    print(f'reply2 결과: {result.get("reply2", "N/A")} \n')

except Exception as e:
    # 어떤 단계에서든 오류가 나면 내용을 출력합니다.
    print(f"Error: {e}")
