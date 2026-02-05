import streamlit as st
import pandas as pd
import ast
import plotly.express as px

# ----------------------------------------------------
# CONFIG & LOAD
# ----------------------------------------------------
st.set_page_config(layout="wide", page_title="Learner Story Dashboard")
INPUT_CSV = "GOV_Reports/gov_story_report.csv"

import json

@st.cache_data
def load_data():
    def safe_parse(val):
        if pd.isna(val): return {}
        s = str(val).strip()
        if not (s.startswith("{") or s.startswith("[")): return {}
        try:
            return ast.literal_eval(s)
        except:
            try:
                # Handle JSON null/true/false
                return json.loads(s)
            except:
                return {}

    try:
        df = pd.read_csv(INPUT_CSV)
        # Parse Dictionary Columns
        for col in ["MD Accuracy Table", "Practice Accuracy Table", "Exit Accuracy Table", "MD Op Details", "Practice Op Details", "Skill History"]:
            if col in df.columns:
                df[col] = df[col].apply(safe_parse)
        
        # Calculate Normalized Gain: (Post - Pre) / (100 - Pre)
        # We use Final Learning Accuracy as Post and MD Accuracy as Pre
        def calc_norm_gain(row):
            pre = row["MD Accuracy"]
            post = row["Final Learning Accuracy"]
            if pd.isna(pre) or pd.isna(post): return 0
            if pre >= 100: return 0 # No room for improvement
            gain = (post - pre) / (100 - pre)
            return round(gain * 100, 1) # Return as percentage

        df["Normalized Gain"] = df.apply(calc_norm_gain, axis=1)
        return df
    except FileNotFoundError:
        st.error(f"File not found: {INPUT_CSV}. Please run 'gov_story.py' first.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# ----------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------
st.sidebar.header("Filters")

def multiselect_filter(label, col):
    if col not in df.columns: return []
    options = sorted(df[col].dropna().unique().astype(str))
    return st.sidebar.multiselect(label, options)

selected_class = multiselect_filter("Class", "Class")
selected_type = multiselect_filter("Student Type", "Student Type")

# Apply Filters
df_filt = df.copy()

# The multiselect_filter function handles column existence checks reliably
selected_state = multiselect_filter("State", "State")
if selected_state: df_filt = df_filt[df_filt["State"].astype(str).isin(selected_state)]

selected_district = multiselect_filter("District", "District")
if selected_district: df_filt = df_filt[df_filt["District"].astype(str).isin(selected_district)]

selected_mandal = multiselect_filter("Mandal", "Mandal")
if selected_mandal: df_filt = df_filt[df_filt["Mandal"].astype(str).isin(selected_mandal)]

selected_school = multiselect_filter("School", "School")
if selected_school: df_filt = df_filt[df_filt["School"].astype(str).isin(selected_school)]

if selected_class: df_filt = df_filt[df_filt["Class"].astype(str).isin(selected_class)]
if selected_type: df_filt = df_filt[df_filt["Student Type"].astype(str).isin(selected_type)]

# ----------------------------------------------------
# MAIN METRICS
# ----------------------------------------------------
st.title("Learner Story Analysis Dashboard")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Learners", len(df_filt))
avg_md = df_filt["MD Accuracy"].mean()
c2.metric("Avg MD Acc", f"{avg_md:.1f}%")
avg_prac = df_filt["Final Learning Accuracy"].mean()
c3.metric("Avg Practice Acc", f"{avg_prac:.1f}%")
avg_imp = df_filt["Improvement"].mean()
c4.metric("Avg Improvement", f"{avg_imp:.1f}%")
avg_norm_gain = df_filt["Normalized Gain"].mean()
c5.metric("Avg Normalized Gain", f"{avg_norm_gain:.1f}%")
c6.metric("Avg Sessions", f"{df_filt['Sessions'].mean():.1f}")

# Extra Row for Location KPIs
st.markdown("#### 🏆 Top Performing Locations")
loc_c1, loc_c2, loc_c3 = st.columns(3)

def get_top_loc(df_in, col):
    # Filter for at least 5 students to avoid local outliers
    stats = df_in.groupby(col).agg(count=('Username', 'count'), gain=('Normalized Gain', 'mean'))
    stats = stats[stats['count'] >= 5].sort_values('gain', ascending=False)
    if not stats.empty:
        return f"{stats.index[0]} ({stats['gain'].iloc[0]:.1f}%)"
    return "N/A"

loc_c1.metric("Top District", get_top_loc(df_filt, "District"))
loc_c2.metric("Top Mandal", get_top_loc(df_filt, "Mandal"))
loc_c3.metric("Top School", get_top_loc(df_filt, "School"))

# ----------------------------------------------------
# TABS
# ----------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Top Learners", "🗺️ Geographic Analysis", "📊 Demographics", "👤 Single Learner", "📘 Learner Story"])

with tab1:
    st.subheader("Top Learner Lists")
    
    # Pre-filters based on gov_story.py logic
    active_learners = df_filt[df_filt["Sessions"] > 20]
    valid_baseline = active_learners[active_learners["MD Accuracy"] > 15]
    avg_attempts = active_learners["Total Attempts"].mean() if not active_learners.empty else 0

    col_a, col_b = st.columns(2)
    with col_a:
        # 1. Improvement in Accuracy
        st.markdown("### 📈 Top Improvers")
        st.caption("MD > 15%, Sessions > 20, Final > 75%, Improvement > 40%")
        top_imp = valid_baseline[
            (valid_baseline["Improvement"] > 40) & 
            (valid_baseline["Final Learning Accuracy"] > 75)
        ].sort_values("Improvement", ascending=False).head(20)
        st.dataframe(top_imp[["Username", "Student Name", "Class", "District", "MD Accuracy", "Final Learning Accuracy", "Improvement", "Sessions"]], use_container_width=True)

    with col_b:
        # 2. Level Jump
        st.markdown("### 🚀 Top Level Jumpers")
        st.caption("MD > 15%, Sessions > 20, Level Jump > 0")
        top_jump = valid_baseline[valid_baseline["Level Jump"] > 0].sort_values("Level Jump", ascending=False).head(20)
        st.dataframe(top_jump[["Username", "Student Name", "Class", "District", "Start Level", "Reached Level", "Level Jump", "Improvement"]], use_container_width=True)

    st.markdown("---")
    
    col_c, col_d = st.columns(2)
    with col_c:
        # 3. Strugglers
        st.markdown("### ⚠️ Top Strugglers")
        st.caption("Sessions > 20, High Attempts (>avg), Jump <= 0, Imp <= 5%")
        strugglers = active_learners[
            (active_learners["Total Attempts"] > avg_attempts) &
            (active_learners["Level Jump"] <= 0) &
            (active_learners["Improvement"] <= 5)
        ].sort_values("Total Attempts", ascending=False).head(20)
        st.dataframe(strugglers[["Username", "Student Name", "Class", "Total Attempts", "Level Jump", "Improvement"]], use_container_width=True)

    with col_d:
        st.markdown("### 🧪 Operation-Wise Improvement")
        st.caption("Detailed view of gains per skill and difficulty")
        
        # Local UI for this specific analysis
        col_op1, col_op2 = st.columns(2)
        with col_op1:
            op_filter = st.selectbox("Select Operation", ["Addition (+)", "Subtraction (-)", "Multiplication (x)", "Division (÷)"], key="local_op_sel")
        with col_op2:
            diff_level = st.selectbox("Difficulty Level", ["Overall", 1, 2, 3, 4, 5], index=0, key="local_diff_sel")
        
        op_map = {"Addition (+)": "+", "Subtraction (-)": "-", "Multiplication (x)": "×", "Division (÷)": "÷"}
        target_op = op_map[op_filter]
        
        def get_op_score(row, phase_col, target_cls, target_op):
            details = row[phase_col]
            if not isinstance(details, dict): return None
            
            # If "Overall", average all valid levels in the dict for this operation
            if target_cls == "Overall":
                scores = []
                for lv_key, val_str in details.items():
                    parts = val_str.split(", ")
                    for p in parts:
                        if target_op in p:
                            try: scores.append(float(p.split(": ")[1].replace("%", "")))
                            except: pass
                return sum(scores) / len(scores) if scores else None
            
            # Specific Level
            key = f"Class {target_cls}" if not str(target_cls).lower().startswith("class") else target_cls
            val_str = details.get(key, "")
            if not val_str: return None
            parts = val_str.split(", ")
            for p in parts:
                if target_op in p:
                    try: return float(p.split(": ")[1].replace("%", ""))
                    except: return None
            return None

        # Calculate for op view
        df_op = df_filt.copy()
        df_op["MD_Op_Acc"] = df_op.apply(lambda r: get_op_score(r, "MD Op Details", diff_level, target_op), axis=1)
        df_op["Pr_Op_Acc"] = df_op.apply(lambda r: get_op_score(r, "Practice Op Details", diff_level, target_op), axis=1)
        df_op["Op_Improvement"] = df_op["Pr_Op_Acc"] - df_op["MD_Op_Acc"]
        top_op_imp = df_op.dropna(subset=["Op_Improvement"]).sort_values("Op_Improvement", ascending=False).head(20)
        
        st.dataframe(top_op_imp[["Username", "Student Name", "MD_Op_Acc", "Pr_Op_Acc", "Op_Improvement"]], use_container_width=True)

with tab2:
    st.subheader("Geographic Breakdown")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### District-Wise Improvement")
        dist_grp = df_filt.groupby("District")["Improvement"].mean().sort_values(ascending=False).reset_index()
        st.dataframe(dist_grp, use_container_width=True)
        # Bar chart
        st.plotly_chart(px.bar(dist_grp, x="District", y="Improvement", title="Avg Improvement by District"), use_container_width=True)

    with col2:
        st.write("#### Mandal-Wise Improvement")
        mand_grp = df_filt.groupby("Mandal")["Improvement"].mean().sort_values(ascending=False).head(20).reset_index()
        st.dataframe(mand_grp, use_container_width=True)
    
    st.write("#### School-Wise Analysis")
    sch_grp = df_filt.groupby("School").agg(
        Count=('Username', 'count'),
        Avg_Imp=('Improvement', 'mean'),
        Avg_Norm_Gain=('Normalized Gain', 'mean'),
        Avg_Prac=('Final Learning Accuracy', 'mean')
    ).sort_values("Avg_Norm_Gain", ascending=False).reset_index()
    st.dataframe(sch_grp, use_container_width=True)

with tab3:
    st.subheader("Demographic & Type Analysis")
    
    type_grp = df_filt.groupby("Student Type").agg(
        Count=('Username', 'count'),
        Avg_Imp=('Improvement', 'mean'),
        Avg_Norm_Gain=('Normalized Gain', 'mean'),
        Avg_Prac=('Final Learning Accuracy', 'mean')
    ).sort_values("Avg_Norm_Gain", ascending=False).reset_index()
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(type_grp, use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(type_grp, x="Student Type", y="Avg_Imp", color="Student Type", title="Improvement by Student Type"), use_container_width=True)

with tab4:
    st.subheader("Learner Profile & Data Extraction")
    
    search_user = st.text_input("Search Learner by Username", placeholder="e.g. 20012290849")
    
    if search_user:
        user_row = df[df["Username"].astype(str) == search_user]
        
        if not user_row.empty:
            row = user_row.iloc[0]
            
            # Header
            st.markdown(f"## {row['Student Name']} ({row['Username']})")
            st.markdown(f"**Class:** {row['Class']} | **School:** {row['School']} ({row['School UDISE']})")
            st.markdown(f"**Location:** {row['District']}, {row['Mandal']}")
            
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Improvement", f"{row['Improvement']}%")
            mc2.metric("Level Jump", f"{row['Level Jump']}")
            mc3.metric("Sessions", f"{row['Sessions']}")
            mc4.metric("Total Attempts", f"{row['Total Attempts']}")
            
            st.markdown("---")
            
            sc1, sc2 = st.columns(2)
            with sc1:
                st.info(f"**Started Level:** {row['Start Level']} → **Reached Level:** {row['Reached Level']}")
                st.write("**Accuracy Trend:**")
                st.write(f"- MD: {row['MD Accuracy']}%")
                st.write(f"- Practice: {row['Final Learning Accuracy']}%")
                st.write(f"- Exit MD: {row['Exit MD Accuracy']}%")
                
                st.write("**Sample Mistakes:**")
                st.warning(row['Wrong Example'])
                st.write("**Sample Masteries:**")
                st.success(row['Correct Example'])

            with sc2:
                st.write("**Detailed Breakdowns:**")
                # Tables
                def dict_to_df(d, name):
                    if not d: return pd.DataFrame()
                    return pd.DataFrame(d.items(), columns=["Level", name])

                md_t = dict_to_df(row["MD Accuracy Table"], "MD Acc")
                pr_t = dict_to_df(row["Practice Accuracy Table"], "Prac Acc")
                ex_t = dict_to_df(row["Exit Accuracy Table"], "Exit Acc")
                
                # Merge tables
                if not md_t.empty:
                    merged_t = md_t.merge(pr_t, on="Level", how="outer").merge(ex_t, on="Level", how="outer")
                    st.table(merged_t)
                
                # Op Details
                if row['MD Op Details']:
                    with st.expander("MD Operation Breakdown"):
                        for cls, det in row['MD Op Details'].items():
                            st.write(f"**{cls}:** {det}")
                if row['Practice Op Details']:
                    with st.expander("Practice Operation Breakdown"):
                        for cls, det in row['Practice Op Details'].items():
                            st.write(f"**{cls}:** {det}")

            st.markdown("---")
            st.subheader("📥 Raw History Extraction")
            st.write("Scan `FinalData.csv` to extract all attempts for this learner.")
            
            if st.button("Extract Raw History (Slow)"):
                RAW_FILE = "FinalData.csv"
                try:
                    import os
                    from pathlib import Path
                    
                    if not os.path.exists(RAW_FILE):
                        st.error(f"Raw data file '{RAW_FILE}' not found in root directory.")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Use logic from data.py
                        CHUNK_SIZE = 500_000
                        num_chunks = int(os.path.getsize(RAW_FILE) / (1024 * 1024 * 50)) # rough estimate for progress
                        
                        collected_rows = []
                        
                        # We count chunks for progress
                        chunk_count = 0
                        for chunk in pd.read_csv(RAW_FILE, chunksize=CHUNK_SIZE, dtype=str, low_memory=False):
                            chunk_count += 1
                            status_text.text(f"Scanning chunk {chunk_count}...")
                            
                            # Filter
                            filtered = chunk[chunk["learner_username"] == search_user].copy()
                            if not filtered.empty:
                                collected_rows.append(filtered)
                            
                            # Update progress (cap at 100%)
                            progress_bar.progress(min(chunk_count / 30, 1.0)) # 30 chunks is ~15M rows
                        
                        if collected_rows:
                            final_df = pd.concat(collected_rows)
                            st.success(f"Extracted {len(final_df)} rows.")
                            csv = final_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"learner_{search_user}_history.csv",
                                mime='text/csv',
                            )
                        else:
                            st.warning("No raw history found in the file.")
                except Exception as e:
                    st.error(f"Error during extraction: {e}")
        else:
            st.error("User not found in the summary report.")

with tab5:
    st.subheader("Interactive Learner Story Generator")
    st.write("Visualizing the granular 'Aha!' moments in a student's journey.")
    
    story_user = st.selectbox("Select a Student to tell their Story", options=df_filt["Username"].unique(), key="story_user_sel")
    
    if story_user:
        s_row = df[df["Username"].astype(str) == story_user].iloc[0]
        
        st.markdown(f"## 📖 The Story of {s_row['Student Name']}")
        
        # 1. The Starting Point
        st.markdown("### 🏃 The Starting Block")
        st.write(f"In Grade {s_row['Class']}, {s_row['Student Name']} started their journey at **Difficulty Level {s_row['Start Level']}**.")
        
        # 2. Level Progression (The Journey)
        st.markdown("### 🗺️ The Learning Journey")
        jcol1, jcol2 = st.columns(2)
        with jcol1:
            st.info(f"**Level Jump:** {s_row['Level Jump']}")
            st.write(f"They progressed from level {s_row['Start Level']} all the way to level {s_row['Reached Level']}.")
        
        # 3. The "Aha!" Moment (Breakthrough Skill)
        if "Breakthrough Skill" in s_row and s_row["Breakthrough Skill"] != "N/A":
            st.markdown("### 💡 The 'Aha!' Moment")
            st.success(f"**Biggest Breakthrough:** {s_row['Breakthrough Skill']}")
            st.write(f"On this specific skill, they improved their accuracy by **{s_row['Breakthrough Gain']}%**!")
            
            # Detailed breakdown of this skill
            history = s_row.get("Skill History", {})
            if history and s_row["Breakthrough Skill"] in history:
                h = history[s_row["Breakthrough Skill"]]
                st.write(f"- Diagnostic Accuracy: {h['md']}%")
                st.write(f"- Practice Accuracy: {h['pr']}%")
        
        # 4. Narrative
        st.markdown("### ✍️ Narrative Analysis")
        
        # Simple rule-based narrative
        narrative = []
        if s_row['Improvement'] > 30:
            narrative.append(f"**{s_row['Student Name']}** showed remarkable resilience. Starting from a diagnostic accuracy of {s_row['MD Accuracy']}%, they pushed through {s_row['Sessions']} sessions to reach a final accuracy of {s_row['Final Learning Accuracy']}%.")
        else:
            narrative.append(f"**{s_row['Student Name']}** showed steady effort over {s_row['Sessions']} sessions, making {s_row['Total Attempts']} attempts to master the material.")
            
        if s_row['Level Jump'] > 1:
            narrative.append(f"Their greatest strength was their fast progression, jumping {s_row['Level Jump']} difficulty levels during their practice period.")
            
        if "Breakthrough Skill" in s_row and s_row["Breakthrough Skill"] != "N/A":
            narrative.append(f"A key turning point was mastering **{s_row['Breakthrough Skill']}**, where they showed their largest single-concept improvement.")
            
        st.write(" ".join(narrative))
        
        # 5. Visualizing all Skills (Skill Progress Plot)
        st.markdown("### 🧬 Detailed Skill Profile")
        history = s_row.get("Skill History", {})
        if history:
            h_df = pd.DataFrame.from_dict(history, orient='index').reset_index()
            h_df.columns = ["Skill", "MD Acc", "Pr Acc"]
            h_df = h_df.dropna(subset=["MD Acc", "Pr Acc"])
            
            if not h_df.empty:
                # Add Improvement col
                h_df["Improvement"] = h_df["Pr Acc"] - h_df["MD Acc"]
                h_df = h_df.sort_values("Improvement", ascending=False)
                
                # Plot
                fig = px.bar(h_df, x="Skill", y=["MD Acc", "Pr Acc"], 
                             title="Skill-wise Accuracy: Diagnostic vs Practice",
                             barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough comparative data for detailed skill plotting yet.")
