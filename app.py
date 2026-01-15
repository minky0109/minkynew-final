import streamlit as st
import fitz
import re
import requests
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 페이지 설정
st.set_page_config(page_title="문항 유사도 분석기", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #F8F4FF; }
    h1, h2, h3 { color: #6F42C1 !important; }
    div.stButton > button { background-color: #6F42C1; color: white; border-radius: 10px; font-weight: bold; height: 3em; }
    .compare-box { border: 2px solid #E0D4F7; padding: 20px; border-radius: 15px; background-color: white; line-height: 1.8; }
    mark { background-color: #E6E0FF; color: #5A32A3; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 구글 드라이브 다운로드 링크 변환 ---
def get_gdrive_direct_link(url):
    file_id = ""
    patterns = [r'/d/([a-zA-Z0-9_-]+)', r'id=([a-zA-Z0-9_-]+)', r'srcid=([a-zA-Z0-9_-]+)']
    for p in patterns:
        match = re.search(p, url)
        if match:
            file_id = match.group(1)
            break
    if file_id:
        return f'https://drive.google.com/uc?export=download&id={file_id}'
    return url

# --- 텍스트 추출 함수 ---
def extract_problems(content, filename):
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        all_problems = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            lines = page.get_text("text").split('\n')
            current_prob, current_num = "", ""
            for line in lines:
                cleaned = line.strip()
                if not cleaned: continue
                match = re.match(r'^(\d+[\.|\)]|\[\d+\])', cleaned)
                if match:
                    if current_prob and len(current_prob) > 40:
                        all_problems.append({"text": current_prob, "page": page_num + 1, "num": current_num, "source": filename})
                    current_num, current_prob = match.group(1).strip(), cleaned
                else:
                    current_prob = (current_prob + " " + cleaned) if current_prob else cleaned
            if current_prob and len(current_prob) > 40:
                all_problems.append({"text": current_prob, "page": page_num + 1, "num": current_num, "source": filename})
        return all_problems
    except:
        return []

# --- 유사도 산출 로직 ---
def calculate_sim(t1, t2):
    v = TfidfVectorizer(ngram_range=(2, 3), analyzer='char')
    try:
        tfidf = v.fit_transform([t1, t2])
        return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except: return 0

# --- 메인 UI ---
st.title("🟣 문항 유사도 분석기")

# [핵심 수정] 초기값(Default Value) 설정
default_links = """모평_수능, https://drive.google.com/file/d/1kf1dZDTFCfAHM9OSAwqaAXI62ClJ3J-S/view?usp=drive_link
2026 수특 생윤, https://drive.google.com/file/d/1xlcMNaNQIbzA1iLXB9lD6eNYL5LM4_LJ/view?usp=drive_link"""

with st.sidebar:
    st.header("🔗 기준 DB 등록")
    # value 인자에 초기값을 넣어 고정시킴
    links_input = st.text_area("이름, 구글링크 (한 줄에 하나씩)", value=default_links, height=200)
    st.caption("기본 링크가 로드되었습니다. 추가 링크가 있다면 아래에 더 작성하세요.")

uploaded_file = st.file_uploader("📝 출제 문항 PDF 업로드 (대상)", type="pdf")

if uploaded_file and links_input:
    if st.button("🚀 분석 시작하기"):
        final_results = []
        all_ref_problems = []
        progress_text = st.empty()
        
        lines = [line for line in links_input.split('\n') if ',' in line]
        
        for idx, line in enumerate(lines):
            name, url = line.split(',', 1)
            progress_text.text(f"📥 {name} 데이터를 가져오는 중...")
            direct_url = get_gdrive_direct_link(url.strip())
            try:
                # 구글 드라이브 다운로드 시 발생할 수 있는 보안 경고 처리 등을 위해 직접 요청
                res = requests.get(direct_url, timeout=30)
                if res.status_code == 200:
                    probs = extract_problems(res.content, name.strip())
                    all_ref_problems.extend(probs)
                else:
                    st.warning(f"{name} 다운로드 실패. 공유 권한을 확인하세요.")
            except Exception as e:
                st.error(f"{name} 연결 중 오류 발생")

        target_probs = extract_problems(uploaded_file.read(), "업로드")

        if all_ref_problems and target_probs:
            progress_bar = st.progress(0)
            for i, target in enumerate(target_probs):
                progress_text.text(f"🔍 {i+1}번 문항 대조 중...")
                best_score, best_match = 0, None
                for ref in all_ref_problems:
                    score = calculate_sim(target['text'], ref['text'])
                    if score > best_score:
                        best_score, best_match = score, ref
                
                final_results.append({
                    "id": i + 1, 
                    "target": target['text'], 
                    "score": round(best_score*100, 1), 
                    "match": best_match
                })
                progress_bar.progress((i + 1) / len(target_probs))
            
            st.session_state['results'] = final_results
            progress_text.success("✅ 분석이 완료되었습니다!")
        else:
            st.error("분석할 데이터를 충분히 확보하지 못했습니다.")

# 결과 출력 영역
if 'results' in st.session_state:
    st.markdown("---")
    for res in st.session_state['results']:
        score = res['score']
        match = res['match']
        status = "🚨 위험" if score > 65 else "⚠️ 주의" if score > 35 else "✅ 안전"
        icon = "🔴" if score > 65 else "🟡" if score > 35 else "🟢"
        
        info = f"[{match['source']} | {match['page']}p {match['num']}]" if match else "정보 없음"
        
        with st.expander(f"{icon} {status} | {res['id']}번 문항 ({score}%) - {info}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"<div class='compare-box'><b>[대상 문항]</b><hr>{res['target']}</div>", unsafe_allow_html=True)
            if match:
                with c2:
                    st.markdown(f"<div class='compare-box'><b>[매칭 DB 문항]</b><hr>{match['text']}</div>", unsafe_allow_html=True)

