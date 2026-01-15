import streamlit as st
import fitz
import re
import requests
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 페이지 설정 및 보라색 디자인 (동일)
st.set_page_config(page_title="문항 유사도 분석기", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #F8F4FF; }
    h1, h2, h3 { color: #6F42C1 !important; }
    div.stButton > button { background-color: #6F42C1; color: white; border-radius: 10px; font-weight: bold; }
    .compare-box { border: 2px solid #E0D4F7; padding: 20px; border-radius: 15px; background-color: white; min-height: 250px; line-height: 1.8; }
    mark { background-color: #E6E0FF; color: #5A32A3; font-weight: bold; border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)

# --- [핵심] 구글 드라이브 링크를 직접 다운로드 링크로 변환 ---
def get_gdrive_direct_link(url):
    # 공유 링크에서 ID 추출
    file_id = ""
    if 'docs.google.com' in url:
        match = re.search(r'/d/([^/]+)', url)
        if match: file_id = match.group(1)
    elif 'drive.google.com' in url:
        match = re.search(r'id=([^&]+)', url)
        if match: file_id = match.group(1)
    
    if file_id:
        return f'https://drive.google.com/uc?export=download&id={file_id}'
    return url

# --- PDF 텍스트 추출 함수 (바이트 데이터 지원) ---
def extract_problems_from_bytes(content, filename):
    doc = fitz.open(stream=content, filetype="pdf")
    all_problems = []
    noise_keywords = ['학년도', '영역', '확인사항', '유의사항', '성명', '수험번호']

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        lines = page.get_text("text").split('\n')
        current_prob, current_num = "", ""
        for line in lines:
            cleaned_line = line.strip()
            if not cleaned_line: continue
            match = re.match(r'^(\d+[\.|\)]|\[\d+\])', cleaned_line)
            if match:
                if current_prob and len(current_prob) >= 40:
                    if not any(nk in current_prob[:30] for nk in noise_keywords):
                        all_problems.append({"text": current_prob, "page": page_num + 1, "num": current_num, "source": filename})
                current_num, current_prob = match.group(1).strip(), cleaned_line
            else:
                current_prob = (current_prob + " " + cleaned_line) if current_prob else cleaned_line
        if current_prob and len(current_prob) >= 40:
            all_problems.append({"text": current_prob, "page": page_num + 1, "num": current_num, "source": filename})
    return all_problems

# --- 유사도 및 하이라이트 로직 (기존과 동일) ---
def calculate_custom_similarity(text1, text2):
    vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char')
    try:
        tfidf = vectorizer.fit_transform([text1, text2])
        v_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except: v_score = 0
    s1, s2 = re.sub(r'\s+', '', text1), re.sub(r'\s+', '', text2)
    common_len = sum(1 for i in range(len(s1)-5) if s1[i:i+5] in s2)
    ratio_score = (common_len * 1.5) / max(len(s1), 1)
    return min(round(((v_score * 0.4) + (ratio_score * 0.6)) * 100, 1), 100.0)

def highlight_selective(target, reference):
    ref_stripped = re.sub(r'\s+', '', reference)
    to_highlight = [target[i:i+6] for i in range(len(target)-5) if re.sub(r'\s+', '', target[i:i+6]) in ref_stripped]
    result = target
    for chunk in sorted(list(set(to_highlight)), key=len, reverse=True):
        if chunk in result: result = result.replace(chunk, f"[[MS]]{chunk}[[ME]]")
    return result.replace("[[MS]]", "<mark>").replace("[[ME]]", "</mark>").replace("</mark><mark>", "")

# --- UI 레이아웃 ---
st.title("🟣 문항 유사도 분석기 (Drive 연동)")

# 사이드바에서 기준 PDF 링크 관리
with st.sidebar:
    st.header("🔗 기준 PDF 링크 등록")
    st.info("구글 드라이브 '링크가 있는 모든 사용자에게 공개' 파일을 등록하세요.")
    
    # 여러 개의 링크를 넣을 수 있도록 설정 (예시 데이터 포함)
    links_input = st.text_area("파일 이름, 드라이브 주소 (한 줄에 하나씩)", 
                               placeholder="수특_생윤, https://drive.google.com/...",
                               height=200)

# 분석 시작
uploaded_file = st.file_uploader("📝 분석할 대상 PDF 업로드", type="pdf")

if uploaded_file and links_input:
    if st.button("🚀 드라이브 데이터 대조 시작"):
        all_ref_problems = []
        with st.spinner('구글 드라이브에서 기준 문항을 가져오는 중...'):
            lines = links_input.split('\n')
            for line in lines:
                if ',' in line:
                    name, url = line.split(',', 1)
                    direct_url = get_gdrive_direct_link(url.strip())
                    try:
                        response = requests.get(direct_url)
                        if response.status_code == 200:
                            probs = extract_problems_from_bytes(response.content, name.strip())
                            all_ref_problems.extend(probs)
                    except:
                        st.error(f"{name} 파일을 가져오지 못했습니다.")

        target_problems = extract_problems_from_bytes(uploaded_file.read(), "업로드파일")
        
        if all_ref_problems and target_problems:
            final_results = []
            for i, target in enumerate(target_problems):
                best_score, best_match = 0, None
                for ref in all_ref_problems:
                    score = calculate_custom_similarity(target['text'], ref['text'])
                    if score > best_score:
                        best_score, best_match = score, ref
                final_results.append({"id": i + 1, "target": target['text'], "score": best_score, "match": best_match})
            st.session_state['drive_results'] = final_results

# 결과 출력
if 'drive_results' in st.session_state:
    for res in st.session_state['drive_results']:
        score = res['score']
        match = res['match']
        status = "🔴 위험" if score > 65 else "🟡 주의" if score > 35 else "🟢 안전"
        source = f"[{match['source']} | {match['page']}p {match['num']}]" if match else "없음"
        
        with st.expander(f"{status} | {res['id']}번 문항 ({score}%) - {source}"):
            if match:
                c1, c2 = st.columns(2)
                with c1: st.markdown(f"<div class='compare-box'><b>[출제]</b><hr>{highlight_selective(res['target'], match['text'])}</div>", unsafe_allow_html=True)
                with c2: st.markdown(f"<div class='compare-box'><b>[DB]</b><hr>{highlight_selective(match['text'], res['target'])}</div>", unsafe_allow_html=True)