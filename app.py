import streamlit as st
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

# تنظیمات کلی صفحه
st.set_page_config(page_title="CRAFT Layout Optimizer", layout="wide")

# ==========================================
# توابع محاسباتی الگوریتم
# ==========================================
def get_exact_centroids(layout):
    """محاسبه دقیق مرکز ثقل هر دپارتمان بر اساس مختصات سلول‌ها"""
    centroids = {}
    for dept in np.unique(layout):
        if dept == 0:
            continue
        r_coords, c_coords = np.where(layout == dept)
        r_center = np.mean(r_coords) + 0.5
        c_center = np.mean(c_coords) + 0.5
        centroids[dept] = (r_center, c_center)
    return centroids

def calculate_exact_cost(layout, flow_matrix):
    """محاسبه گشتاور/هزینه کل بر اساس فاصله مستطیلی و جریان مواد"""
    centroids = get_exact_centroids(layout)
    total_cost = 0.0
    depts = list(centroids.keys())
    for i in depts:
        for j in depts:
            f = flow_matrix[i-1][j-1]
            if f > 0:
                r1, c1 = centroids[i]
                r2, c2 = centroids[j]
                dist = abs(r1 - r2) + abs(c1 - c2)
                total_cost += f * dist
    return total_cost

def get_distance_matrix(layout, n_depts):
    """تولید ماتریس فاصله‌های مستطیلی بین مراکز ثقل برای نمایش به کاربر"""
    centroids = get_exact_centroids(layout)
    dist_matrix = np.zeros((n_depts, n_depts))
    depts = list(centroids.keys())
    for i in depts:
        for j in depts:
            if i != j:
                r1, c1 = centroids[i]
                r2, c2 = centroids[j]
                dist_matrix[i-1, j-1] = abs(r1 - r2) + abs(c1 - c2)
    return dist_matrix

def is_contiguous(layout, dept_id):
    """بررسی عدم گسستگی (پیوستگی سلول‌های یک دپارتمان) با استفاده از BFS"""
    cells = list(zip(*np.where(layout == dept_id)))
    if not cells:
        return True
    visited = set()
    queue = [cells[0]]
    visited.add(cells[0])
    cell_set = set(cells)
    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cell_set and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return len(visited) == len(cells)

def are_adjacent(layout, d1, d2):
    """بررسی شرط همسایگی بین دو دپارتمان"""
    r1, c1 = np.where(layout == d1)
    r2, c2 = np.where(layout == d2)
    set2 = set(zip(r2, c2))
    for r, c in zip(r1, c1):
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            if (r + dr, c + dc) in set2:
                return True
    return False

def new_craft_algorithm(initial_layout, flow_matrix, enable_3way=True):
    """اجرای الگوریتم توسعه‌یافته CRAFT با جابجایی‌های ۲تایی و ۳تایی"""
    current_layout = initial_layout.copy()
    current_cost = calculate_exact_cost(current_layout, flow_matrix)
    history = [(current_layout.copy(), current_cost, "طرح اولیه")]
    
    while True:
        best_layout = None
        best_cost = current_cost
        best_move_type = ""
        depts = [d for d in np.unique(current_layout) if d != 0]
        
        # بررسی جابجایی‌های دوتایی (2-Way Swaps)
        for d1, d2 in itertools.combinations(depts, 2):
            area_match = (np.sum(current_layout == d1) == np.sum(current_layout == d2))
            adj = are_adjacent(current_layout, d1, d2)
            if area_match or adj:
                temp_layout = current_layout.copy()
                p1 = np.where(temp_layout == d1)
                p2 = np.where(temp_layout == d2)
                temp_layout[p1], temp_layout[p2] = d2, d1
                if is_contiguous(temp_layout, d1) and is_contiguous(temp_layout, d2):
                    exact_cost = calculate_exact_cost(temp_layout, flow_matrix)
                    if exact_cost < best_cost - 1e-5:
                        best_cost = exact_cost
                        best_layout = temp_layout.copy()
                        best_move_type = f"جابجایی ۲تایی ({chr(64+d1)} ↔ {chr(64+d2)})"

        # بررسی جابجایی‌های سه‌تایی (3-Way Swaps)
        if enable_3way:
            for d1, d2, d3 in itertools.permutations(depts, 3):
                c1 = (np.sum(current_layout == d1) == np.sum(current_layout == d2)) or are_adjacent(current_layout, d1, d2)
                c2 = (np.sum(current_layout == d2) == np.sum(current_layout == d3)) or are_adjacent(current_layout, d2, d3)
                c3 = (np.sum(current_layout == d3) == np.sum(current_layout == d1)) or are_adjacent(current_layout, d3, d1)
                
                if c1 and c2 and c3:
                    temp_layout = current_layout.copy()
                    p1 = np.where(current_layout == d1)
                    p2 = np.where(current_layout == d2)
                    p3 = np.where(current_layout == d3)
                    temp_layout[p2], temp_layout[p3], temp_layout[p1] = d1, d2, d3
                    
                    if is_contiguous(temp_layout, d1) and is_contiguous(temp_layout, d2) and is_contiguous(temp_layout, d3):
                        exact_cost = calculate_exact_cost(temp_layout, flow_matrix)
                        if exact_cost < best_cost - 1e-5:
                            best_cost = exact_cost
                            best_layout = temp_layout.copy()
                            best_move_type = f"جابجایی ۳تایی ({chr(64+d1)} → {chr(64+d2)} → {chr(64+d3)})"

        if best_layout is None:
            break
            
        current_layout = best_layout
        current_cost = best_cost
        history.append((current_layout.copy(), current_cost, best_move_type))
        
    return current_layout, current_cost, history

# ==========================================
# تابع رسم گرافیکی چیدمان
# ==========================================
def draw_layout_grid(layout):
    """نمایش بصری چیدمان کارخانه با کتابخانه Matplotlib"""
    fig, ax = plt.subplots(figsize=(layout.shape[1] * 1.5, layout.shape[0] * 1.5))
    ax.matshow(layout, cmap='Pastel1', vmin=0)
    
    for (i, j), val in np.ndenumerate(layout):
        if val > 0:
            dept_name = chr(64 + val)
            ax.text(j, i, dept_name, ha='center', va='center', 
                    fontsize=20, fontweight='bold', color='black')
        else:
            ax.text(j, i, "-", ha='center', va='center', fontsize=20, color='gray')
            
    ax.set_xticks(np.arange(-0.5, layout.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, layout.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

# ==========================================
# رابط کاربری (UI)
# ==========================================
st.title("نرم‌افزار بهینه‌سازی چیدمان کارخانه (مبتنی بر رویکرد توسعه‌یافته CRAFT)")
st.caption("CRAFT-Based Layout Optimization Software (Extended New CRAFT Approach)")

st.info("""
**معرفی الگوریتم:** 
نرم‌افزار ارائه‌شده یک محیط تصمیم‌یار مبتنی بر نسخه توسعه‌یافته الگوریتم CRAFT است که با بررسی حرکات دوگانه و سه‌گانه (2-Way & 3-Way Exchanges)، کنترل پیوستگی دپارتمان‌ها و محاسبه دقیق هزینه جابجایی مواد، چیدمان اولیه کارخانه را بهبود می‌دهد.
""")

# منوی تنظیمات پارامترها
st.sidebar.header("پارامترهای شبکه و دپارتمان‌ها")
n_depts = st.sidebar.number_input("تعداد دپارتمان‌ها", min_value=2, max_value=10, value=4)
rows = st.sidebar.number_input("تعداد سطرهای شبکه‌بندی", min_value=1, value=2)
cols = st.sidebar.number_input("تعداد ستون‌های شبکه‌بندی", min_value=1, value=2)
enable_3way = st.sidebar.checkbox("فعال‌سازی جابجایی‌های ۳‌تایی", value=True)

# کنترل ایمنی ابعاد شبکه برای جلوگیری از باگ
if rows * cols < n_depts:
    st.sidebar.error("خطا: تعداد سلول‌های Layout (سطر × ستون) نمی‌تواند کمتر از تعداد دپارتمان‌ها باشد.")
    st.stop()

st.subheader("۱. ماتریس جریان (Flow Matrix)")

col_names = [str(chr(65+i)) for i in range(n_depts)]

if n_depts == 4:
    raw_data = [[0, 2, 4, 3], [8, 0, 4, 2], [20, 10, 0, 3], [8, 3, 2, 0]]
else:
    raw_data = np.zeros((n_depts, n_depts), dtype=int).tolist()

# ساخت دیتافریم با اجبار به استفاده از تایپ‌های استاندارد پایتون (جلوگیری از خطای LargeUtf8)
default_flow = pd.DataFrame(raw_data, columns=col_names, index=col_names)
default_flow.columns = pd.Index(col_names, dtype=object)
default_flow.index = pd.Index(col_names, dtype=object)
default_flow = default_flow.astype(int)

flow_df = st.data_editor(default_flow, use_container_width=True)

if st.button("اجرای الگوریتم بهینه‌سازی", type="primary"):
    flow_matrix = flow_df.values
    
    # تولید چیدمان اولیه ایمن با پشتیبانی از فضاهای خالی
    initial_layout = np.zeros((rows, cols), dtype=int)
    dept_ids = list(range(1, n_depts + 1))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < n_depts:
                initial_layout[r, c] = dept_ids[idx]
                idx += 1
    
    # اجرای هسته الگوریتم
    final_layout, final_cost, history = new_craft_algorithm(initial_layout, flow_matrix, enable_3way)
    initial_cost = history[0][1]
    
    # محاسبات عملکرد
    improvement_percent = ((initial_cost - final_cost) / initial_cost) * 100 if initial_cost > 0 else 0
    
    st.divider()
    st.subheader("۲. داشبورد نتایج بهینه‌سازی")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("هزینه طرح اولیه (Initial Cost)", f"{initial_cost:.2f}")
    col2.metric("هزینه طرح نهایی (Final Cost)", f"{final_cost:.2f}", f"-{improvement_percent:.1f}% Improvement", delta_color="inverse")
    col3.metric("تعداد گام‌های بهبود (Iterations)", len(history) - 1)
    
    # نمایش ماتریس فاصله برای درک شفاف تابع هدف
# نمایش ماتریس فاصله برای درک شفاف تابع هدف (اصلاح شده برای جلوگیری از خطای LargeUtf8)
    with st.expander("📊 نمایش ماتریس فاصله (Distance Matrix) برای طرح نهایی"):
        st.markdown("این ماتریس فاصله مستطیلی بین مرکز ثقل دپارتمان‌ها در طرح نهایی را نشان می‌دهد:")
        dist_mat = get_distance_matrix(final_layout, n_depts)
        dist_names = [str(chr(65+i)) for i in range(n_depts)]
        
        dist_df = pd.DataFrame(dist_mat, columns=dist_names, index=dist_names)
        dist_df.columns = pd.Index(dist_names, dtype=object)
        dist_df.index = pd.Index(dist_names, dtype=object)
        
        st.dataframe(dist_df, use_container_width=True)
    
    st.divider()
    st.subheader("۳. روند تغییرات چیدمان (Layout Progression)")
    
    # نمایش گام‌به‌گام بهبودها
    for idx, (lyt, cst, move) in enumerate(history):
        col_text, col_plot = st.columns([1, 2])
        
        with col_text:
            st.markdown(f"### گام {idx}")
            st.markdown(f"**عملیات:** {move}")
            st.markdown(f"**هزینه:** {cst:.2f}")
            
        with col_plot:
            fig = draw_layout_grid(lyt)
            st.pyplot(fig)