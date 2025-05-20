from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import os

from pinecone import Pinecone, PodSpec
from tqdm.auto import tqdm # 진행률 표시를 위한 라이브러리 (설치 필요: pip install tqdm)

# 1. 환경 변수 로드
load_dotenv()

# 2. 문서 로더 및 분할기 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # 각 청크의 최대 크기
    chunk_overlap=200, # 청크 간 겹치는 부분의 크기
)

# Docx 파일 로드 및 분할
try:
    loader = Docx2txtLoader('./tax_markdown-index.docx')
    document_list = loader.load_and_split(text_splitter=text_splitter)
    print(f"총 {len(document_list)}개의 문서 청크를 로드했습니다.")
except Exception as e:
    print(f"문서 로드 중 오류가 발생했습니다: {e}")
    print("파일 경로('./tax_markdown-index.docx')가 올바른지, 파일이 손상되지 않았는지 확인해주세요.")
    exit() # 오류 발생 시 스크립트 종료

# 3. OpenAI 임베딩 모델 설정
embedding = OpenAIEmbeddings(model='text-embedding-3-large')

# 4. Pinecone 설정
index_name = 'tax-markdown-index'
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

if not pinecone_api_key:
    print("PINECONE_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    exit()

pc = Pinecone(api_key=pinecone_api_key)

# 5. Pinecone 인덱스 존재 여부 확인 및 생성
existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:
# --- 수정된 부분 끝 ---
    print(f"'{index_name}' 인덱스가 존재하지 않아 새로 생성합니다.")
    try:
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric='cosine',
            spec=PodSpec(environment="aws-us-east-1T")
        )
        print(f"'{index_name}' 인덱스 생성을 완료했습니다.")
    except Exception as e:
        print(f"인덱스 생성 중 오류가 발생했습니다: {e}")
        print("Pinecone 환경(environment) 설정이 올바른지, API 키가 유효한지 확인해주세요.")
        exit()
else:
    print(f"'{index_name}' 인덱스가 이미 존재합니다. 기존 인덱스를 사용합니다.")

# 6. Pinecone Vector Store 초기화
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding)

# 7. 문서들을 배치로 나누어 Pinecone에 업로드
batch_size = 100 # 한 번에 Pinecone에 전송할 문서 청크의 개수

print(f"문서들을 {batch_size}개씩 배치로 나누어 Pinecone에 업로드합니다...")
for i in tqdm(range(0, len(document_list), batch_size), desc="Pinecone 업로드 진행률"):
    i_end = min(i + batch_size, len(document_list))
    batch = document_list[i:i_end]
    try:
        vectorstore.add_documents(batch)
    except Exception as e:
        print(f"배치 업로드 중 오류가 발생했습니다 (인덱스 {i}~{i_end-1}): {e}")
        print("Batch size를 더 작게 조절해보거나, 오류 메시지를 다시 확인해주세요.")

print("\n모든 문서 청크가 Pinecone에 성공적으로 업로드되었거나 처리되었습니다.")