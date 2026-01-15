import streamlit as st
import fitz
import re
import requests
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 페이지 설정 및 디자인
st.set_page_config(page_title="문항 유사도 분석기", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #F8F4FF; }
    h1, h2, h3 { color: #6F42C1 !important; }
    div.stButton > button { background-color: #6F42C1; color: white; border-radius: 10px; font-weight: bold; height: 3.5em; border: none; }
    div.stButton > button:hover { background-color: #5A32A3; color: white; }
    .compare-box { border: 2px solid #E0D4F7; padding: 20px; border-radius: 15px; background-color: white; line-height: 1.8; overflow-wrap: break-word; min-height: 200px; }
    mark { background-color: #E6E0FF; color: #5A32A3; font-weight: bold; border-radius: 3px; padding: 0 2px; }
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
    return f'https://drive.google.com/uc?export=download&id={file_id}' if file_id else url

# --- 텍스트 추출 및 문항 분별 ---
def extract_problems_refined(content, filename):
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        all_problems = []
        skip_keywords = ['학년도', '영역', '확인사항', '유의사항', '성명', '수험번호', '문제지', '탐구']

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            lines = page.get_text("text").split('\n')
            current_prob, current_num = "", ""

            for line in lines:
                cleaned = line.strip()
                if not cleaned or len(cleaned) < 2: continue
                if any(kw in cleaned for kw in skip_keywords): continue

                num_match = re.match(r'^(\d+[\.|\)]|\[\d+\])', cleaned)
                if num_match:
                    if current_prob and len(current_prob) > 30:
                        all_problems.append({"text": current_prob, "page": page_num + 1, "num": current_num if current_num else "미상", "source": filename})
                    current_num, current_prob = num_match.group(1).strip(), cleaned
                else:
                    if current_prob: current_prob += " " + cleaned
            
            if current_prob and len(current_prob) > 30:
                all_problems.append({"text": current_prob, "page": page_num + 1, "num": current_num if current_num else "미상", "source": filename})
        return all_problems
    except: return []

# --- 유사도 산출 ---
def calculate_sim(t1, t2):
    v = TfidfVectorizer(ngram_range=(2, 4), analyzer='char')
    try:
        tfidf = v.fit_transform([t1, t2])
        return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except: return 0

# --- [핵심 추가] 하이라이팅 로직 ---
def highlight_overlap(target, reference):
    if not target or not reference: return target
    
    # 공백 제거 후 비교를 위해 정규화
    ref_clean = re.sub(r'\s+', '', reference)
    
    # 6글자씩 슬라이딩 윈도우로 겹치는 구간 탐색
    min_len = 6
    to_highlight = []
    for i in range(len(target) - min_len + 1):
        chunk = target[i:i+min_len]
        if len(chunk.strip()) < min_len: continue
        if re.sub(r'\s+', '', chunk) in ref_clean:
            to_highlight.append(chunk)
    
    # 겹치는 구간들을 병합하여 mark 태그 삽입
    sorted_chunks = sorted(list(set(to_highlight)), key=len, reverse=True)
    result = target
    for chunk in sorted_chunks:
        if chunk in result:
            result = result.replace(chunk, f"[[MS]]{chunk}[[ME]]")
    
    # 중복 태그 정리 및 치환
    result = result.replace("[[MS]]", "<mark>").replace("[[ME]]", "</mark>")
    result = re.sub(r'(</mark>\s*<mark>)', '', result)
    return result

# --- UI 레이아웃 ---
st.title("🟣 문항 유사도 분석기")

default_links = """모평_수능, https://drive.google.com/file/d/1kf1dZDTFCfAHM9OSAwqaAXI62ClJ3J-S/view?usp=drive_link
2026 수특 생윤, https://drive.google.com/file/d/1xlcMNaNQIbzA1iLXB9lD6eNYL5LM4_LJ/view?usp=drive_link"""

with st.sidebar:
    st.header("🔗 기준 DB 등록")
    links_input = st.text_area("이름, 구글링크", value=default_links, height=150)

uploaded_file = st.file_uploader("📝 분석할 문항 PDF 업로드", type="pdf")

if uploaded_file and links_input:
    if st.button("🚀 정밀 분석 및 하이라이팅 시작"):
        final_results = []
        all_ref_problems = []
        
        # 1. DB 로드
        lines = [line for line in links_input.split('\n') if ',' in line]
        for line in lines:
            name, url = line.split(',', 1)
            direct_url = get_gdrive_direct_link(url.strip())
            try:
                res = requests.get(direct_url, timeout=30)
                if res.status_code == 200:
                    all_ref_problems.extend(extract_problems_refined(res.content, name.strip()))
            except: pass

        # 2. 업로드 파일 로드
        target_probs = extract_problems_refined(uploaded_file.read(), "업로드")

        # 3. 분석
        if all_ref_problems and target_probs:
            prog = st.progress(0)
            status = st.empty()
            for i, target in enumerate(target_probs):
                t_num = target.get('num', '미상')
                status.text(f"🔍 {i+1}번({t_num}) 문항 분석 중...")
                best_score, best_match = 0, None
                for ref in all_ref_problems:
                    score = calculate_sim(target['text'], ref['text'])
                    if score > best_score:
                        best_score, best_match = score, ref
                
                final_results.append({
                    "id": i + 1, "target": target.get('text', ''), "num": t_num,
                    "score": round(best_score*100, 1), "match": best_match
                })
                prog.progress((i + 1) / len(target_probs))
            st.session_state['results'] = final_results
            status.success("✅ 정밀 분석 완료!")

# --- 결과 표시 ---
if 'results' in st.session_state:
    st.markdown("---")
    for res in st.session_state['results']:
        score = res.get('score', 0)
        num = res.get('num', '미상')
        color = "🔴" if score > 65 else "🟡" if score > 35 else "🟢"
        match = res.get('match')
        
        info = f" - [매칭: {match['source']} {match['page']}p {match['num']}]" if match else ""
        label = f"{color} {num}번 (유사도 {score}%){info}"

        with st.expander(label):
            c1, c2 = st.columns(2)
            if match:
                # 하이라이팅 적용
                h_target = highlight_overlap(res['target'], match['text'])
                h_match = highlight_overlap(match['text'], res['target'])
                
                with c1: st.markdown(f"<div class='compare-box'><b>[대상 문항]</b><hr>{h_target}</div>", unsafe_allow_html=True)
                with c2: st.markdown(f"<div class='compare-box'><b>[DB 문항]</b><hr>{h_match}</div>", unsafe_allow_html=True)
            else:
                with c1: st.markdown(f"<div class='compare-box'><b>[대상 문항]</b><hr>{res['target']}</div>", unsafe_allow_html=True)
                with c2: st.info("매칭된 데이터가 없습니다.")
