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
    div.stButton > button { background-color: #6F42C1; color: white; border-radius: 10px; font-weight: bold; height: 3.5em; }
    .compare-box { border: 2px solid #E0D4F7; padding: 20px; border-radius: 15px; background-color: white; line-height: 1.8; }
    mark { background-color: #E6E0FF; color: #5A32A3; font-weight: bold; padding: 0 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- 구글 드라이브 링크 변환 ---
def get_gdrive_direct_link(url):
    file_id = ""
    patterns = [r'/d/([a-zA-Z0-9_-]+)', r'id=([a-zA-Z0-9_-]+)', r'srcid=([a-zA-Z0-9_-]+)']
    for p in patterns:
        match = re.search(p, url)
        if match:
            file_id = match.group(1); break
    return f'https://drive.google.com/uc?export=download&id={file_id}' if file_id else url

# --- [정밀도 극대화] 좌표 기반 텍스트 재정렬 및 추출 ---
def extract_problems_refined(content, filename):
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        all_problems = []
        skip_keywords = ['학년도', '영역', '확인사항', '유의사항', '성명', '수험번호', '문제지', '탐구', '사회·문화', '생활과 윤리']
        
        current_prob = ""
        current_num = ""
        current_page = 1

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # 텍스트를 개별 단어 단위로 좌표와 함께 추출
            words = page.get_text("words") 
            # 1. Y좌표(높이)로 1차 정렬, 2. X좌표(가로)로 2차 정렬하여 인간의 독서 순서 재현
            words.sort(key=lambda w: (w[1], w[0])) 

            # 단어들을 줄 단위로 묶기
            lines = []
            if words:
                last_y = words[0][1]
                current_line = []
                for w in words:
                    # Y좌표 차이가 작으면 같은 줄로 인식 (오차 범위 3포인트)
                    if abs(w[1] - last_y) < 3:
                        current_line.append(w[4])
                    else:
                        lines.append(" ".join(current_line))
                        current_line = [w[4]]
                        last_y = w[1]
                lines.append(" ".join(current_line))

            for line_text in lines:
                cleaned = line_text.strip()
                if not cleaned or len(cleaned) < 2: continue
                if any(kw in cleaned for kw in skip_keywords): continue

                # 문항 번호 감지 (강력한 패턴: 숫자 뒤 점/괄호/대괄호)
                num_match = re.match(r'^(\d+[\.|\)]|\[\d+\])', cleaned)
                
                if num_match:
                    if current_prob.strip():
                        all_problems.append({
                            "text": current_prob.strip(),
                            "page": current_page,
                            "num": current_num if current_num else "미상",
                            "source": filename
                        })
                    current_num = num_match.group(1).strip()
                    current_prob = cleaned
                    current_page = page_num + 1
                else:
                    if current_prob:
                        current_prob += " " + cleaned
                    else:
                        current_prob = cleaned
                        current_page = page_num + 1

        # 마지막 문항 수집
        if current_prob.strip():
            all_problems.append({
                "text": current_prob.strip(),
                "page": current_page,
                "num": current_num if current_num else "마지막",
                "source": filename
            })
            
        return all_problems
    except Exception as e:
        return []

# --- 하이라이팅 및 분석 로직 (동일) ---
def highlight_overlap(target, reference):
    if not target or not reference: return target
    ref_clean = re.sub(r'\s+', '', reference)
    min_len = 6
    to_highlight = []
    for i in range(len(target) - min_len + 1):
        chunk = target[i:i+min_len]
        if len(chunk.strip()) < min_len: continue
        if re.sub(r'\s+', '', chunk) in ref_clean: to_highlight.append(chunk)
    
    sorted_chunks = sorted(list(set(to_highlight)), key=len, reverse=True)
    result = target
    for chunk in sorted_chunks:
        if chunk in result: result = result.replace(chunk, f"[[MS]]{chunk}[[ME]]")
    return result.replace("[[MS]]", "<mark>").replace("[[ME]]", "</mark>").replace("</mark><mark>", "")

# --- 메인 실행부 (고정 링크 포함) ---
st.title("🟣 문항 유사도 분석기")

default_links = """모평_수능, https://drive.google.com/file/d/1kf1dZDTFCfAHM9OSAwqaAXI62ClJ3J-S/view?usp=drive_link
2026 수특 생윤, https://drive.google.com/file/d/1xlcMNaNQIbzA1iLXB9lD6eNYL5LM4_LJ/view?usp=drive_link
사문_모평, https://drive.google.com/file/d/1QTIRXZdqlixqhLlUsywqGHZcrxdqZ_mN/view?usp=sharing
2026 사문_수특, https://drive.google.com/file/d/1V-WjvOsOSZwuuRaRObwPqdD07Rvuyx7f/view?usp=drive_link"""

with st.sidebar:
    st.header("🔗 기준 DB 등록")
    links_input = st.text_area("이름, 구글링크", value=default_links, height=200)

uploaded_file = st.file_uploader("📝 분석할 문항 PDF 업로드", type="pdf")

if uploaded_file and links_input:
    if st.button("🚀 정밀 분석 시작"):
        final_results = []
        all_ref_problems = []
        status_msg = st.empty()
        
        session = requests.Session()
        lines = [line for line in links_input.split('\n') if ',' in line]
        
        for line in lines:
            name, url = line.split(',', 1)
            name = name.strip()
            status_msg.info(f"⏳ '{name}' 데이터를 분석용으로 재구성 중...")
            direct_url = get_gdrive_direct_link(url.strip())
            try:
                res = session.get(direct_url, timeout=60)
                if res.status_code == 200:
                    all_ref_problems.extend(extract_problems_refined(res.content, name))
            except: pass

        if all_ref_problems:
            target_probs = extract_problems_refined(uploaded_file.read(), "업로드")
            
            if target_probs:
                prog = st.progress(0)
                vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char')
                
                for i, target in enumerate(target_probs):
                    t_num = target.get('num', '미상')
                    status_msg.info(f"🔍 {i+1}번({t_num}) 문항 대조 중...")
                    
                    best_score, best_match = 0, None
                    for ref in all_ref_problems:
                        try:
                            tfidf = vectorizer.fit_transform([target['text'], ref['text']])
                            score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                            if score > best_score:
                                best_score, best_match = score, ref
                        except: continue
                    
                    final_results.append({
                        "id": i + 1, "target": target['text'], "num": t_num,
                        "score": round(best_score*100, 1), "match": best_match
                    })
                    prog.progress((i + 1) / len(target_probs))
                
                st.session_state['results'] = final_results
                status_msg.success(f"✅ 총 {len(target_probs)}개 문항 정렬 분석 완료!")
