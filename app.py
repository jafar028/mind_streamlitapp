import datetime as dt
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account

PROJECT = "gen-lang-client-0625543859"
DATASET = f"{PROJECT}.mind_documentdb"
LOCAL_KEY = Path(__file__).parent / "gen-lang-client-0625543859-65541d36d0ea.json"

st.set_page_config(page_title="MIND Analytics", page_icon="📊", layout="wide")


@st.cache_resource
def get_client() -> bigquery.Client:
    """Credentials from st.secrets (deployed), GOOGLE_APPLICATION_CREDENTIALS, or the local key file."""
    try:
        info = dict(st.secrets["gcp_service_account"])
    except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        info = None
    if info:
        creds = service_account.Credentials.from_service_account_info(info)
    else:
        key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", str(LOCAL_KEY))
        creds = service_account.Credentials.from_service_account_file(key)
    return bigquery.Client(credentials=creds, project=PROJECT)


def run_query(sql: str, params: list | None = None) -> pd.DataFrame:
    job_config = bigquery.QueryJobConfig(query_parameters=params or [])
    return get_client().query(sql, job_config=job_config).to_dataframe(create_bqstorage_client=False)


# ---------------------------------------------------------------- sessions by email

SESSIONS_SQL = f"""
SELECT
  u.email, s.room_name, s.end_reason, s.end_time, s.start_time, s.is_active
FROM `{DATASET}.users` u
JOIN `{DATASET}.sessions` s
  ON u._id = s.user_id
WHERE LOWER(u.email) = LOWER(@email)
ORDER BY s.start_time DESC
"""


@st.cache_data(ttl=300, show_spinner="Fetching sessions…")
def get_sessions(email: str) -> pd.DataFrame:
    return run_query(SESSIONS_SQL, [bigquery.ScalarQueryParameter("email", "STRING", email)])


# ---------------------------------------------------------------- case study coverage

PROGRAMME_CODES = [
    "MIT 8113", "MIT 8208", "MIT 8307", "MIT 8319", "MIT 8349", "MIT 8345",
    "MPA 8107", "MPA 8204", "MPA 8323", "MPA 8327", "MPA 8309", "MPA 8315",
    "MPH 8426", "MPH 8404", "MPH 8408", "MPH 8444", "MPH 8462",
    "MBA 8111", "MBA 8206", "MBA 8307", "MBA 8402",
    "MID 8105", "MCS 8111", "IFT 410", "MIVA-AIT 101", "MIVA-CSS 408", "ENT 424", "MCM 405",
    "MIVA-PAD 102", "EHS 104", "MIVA-CHS 104", "MIVA-EEC 105", "PED 111", "MIVA-PED 119",
    "DBA 9101", "DBA 9103", "DBA 9105",
]

COVERAGE_SQL = f"""
WITH user_programmes AS (
  SELECT
    _id AS user_id,
    email,
    programme_code
  FROM `{DATASET}.users`,
  UNNEST(JSON_EXTRACT_STRING_ARRAY(programme_codes)) AS programme_code
  WHERE email LIKE '%miva.edu.ng%'
    AND email NOT LIKE '%gmail.com%'
    AND NOT REGEXP_CONTAINS(UPPER(programme_code), r'DBA')
    AND programme_code IN UNNEST(@programme_codes)
),

case_study_programmes AS (
  SELECT
    _id AS case_study_id,
    title,
    programme_code
  FROM `{DATASET}.case_studies`,
  UNNEST(JSON_EXTRACT_STRING_ARRAY(programme_codes)) AS programme_code
  WHERE created_at >= @case_studies_since
),

attempts AS (
  SELECT user_id, content_id, COUNT(*) AS attempt_count
  FROM `{DATASET}.grades`
  WHERE attempt_started_at >= @attempts_since
  GROUP BY user_id, content_id
)

SELECT
  up.email,
  up.programme_code AS course_code,
  csp.title AS case_study_title,
  a.user_id IS NOT NULL AS has_taken,
  COALESCE(a.attempt_count, 0) AS attempt_count
FROM user_programmes up
JOIN case_study_programmes csp
  ON up.programme_code = csp.programme_code
LEFT JOIN attempts a
  ON a.user_id = up.user_id
 AND a.content_id = csp.case_study_id
ORDER BY up.email, course_code, case_study_title
"""


@st.cache_data(ttl=600, show_spinner="Crunching case study coverage…")
def get_coverage(case_studies_since: dt.date, attempts_since: dt.date) -> pd.DataFrame:
    params = [
        bigquery.ScalarQueryParameter("case_studies_since", "TIMESTAMP", dt.datetime.combine(case_studies_since, dt.time())),
        bigquery.ScalarQueryParameter("attempts_since", "TIMESTAMP", dt.datetime.combine(attempts_since, dt.time())),
        bigquery.ArrayQueryParameter("programme_codes", "STRING", PROGRAMME_CODES),
    ]
    # One row per (email, course, case study); collapse any duplicate titles.
    df = run_query(COVERAGE_SQL, params)
    return (
        df.groupby(["email", "course_code", "case_study_title"], as_index=False)
        .agg(has_taken=("has_taken", "max"), attempt_count=("attempt_count", "sum"))
    )


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["course_code", "case_study_title"], as_index=False)
        .agg(
            students=("email", "nunique"),
            attempted=("has_taken", "sum"),
            total_attempts=("attempt_count", "sum"),
        )
        .assign(
            not_attempted=lambda d: d["students"] - d["attempted"],
            completion_rate=lambda d: d["attempted"] / d["students"],
        )
    )


STATUS_COLORS = alt.Scale(domain=["Attempted", "Not attempted"], range=["#2a9d8f", "#e76f51"])


def coverage_chart(summary: pd.DataFrame, y_field: str, y_title: str) -> alt.Chart:
    long = summary.melt(
        id_vars=[y_field, "students", "total_attempts", "completion_rate"],
        value_vars=["attempted", "not_attempted"],
        var_name="status",
        value_name="users",
    )
    long["status"] = long["status"].map({"attempted": "Attempted", "not_attempted": "Not attempted"})
    order = summary.sort_values("completion_rate")[y_field].tolist()
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            y=alt.Y(f"{y_field}:N", sort=order, title=y_title, axis=alt.Axis(labelLimit=320)),
            x=alt.X("users:Q", title="Students", stack="zero"),
            color=alt.Color("status:N", scale=STATUS_COLORS, legend=alt.Legend(title=None, orient="top")),
            order=alt.Order("status:N"),
            tooltip=[
                alt.Tooltip(f"{y_field}:N", title=y_title),
                alt.Tooltip("status:N", title="Status"),
                alt.Tooltip("users:Q", title="Students"),
                alt.Tooltip("students:Q", title="Enrolled"),
                alt.Tooltip("total_attempts:Q", title="Total attempts"),
                alt.Tooltip("completion_rate:Q", title="Completion", format=".0%"),
            ],
        )
        .properties(height=max(160, 42 * summary[y_field].nunique()))
    )


# ---------------------------------------------------------------- UI

st.title("📊 MIND Analytics")

tab_sessions, tab_coverage = st.tabs(["🔎 Sessions by email", "📚 Case study attempts"])

with tab_sessions:
    with st.form("email_lookup"):
        email = st.text_input("Student email", placeholder="name@miva.edu.ng")
        submitted = st.form_submit_button("Search", type="primary")

    if submitted:
        email = email.strip()
        if not email:
            st.warning("Enter an email address.")
        else:
            sessions = get_sessions(email)
            if sessions.empty:
                st.info(f"No sessions found for **{email}**.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Sessions", len(sessions))
                c2.metric("Active now", int(sessions["is_active"].fillna(False).sum()))
                c3.metric("Rooms", sessions["room_name"].nunique())
                st.dataframe(
                    sessions,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "start_time": st.column_config.DatetimeColumn("Start time", format="YYYY-MM-DD HH:mm"),
                        "end_time": st.column_config.DatetimeColumn("End time", format="YYYY-MM-DD HH:mm"),
                        "is_active": st.column_config.CheckboxColumn("Active"),
                    },
                )
                st.download_button(
                    "Download CSV", sessions.to_csv(index=False), file_name=f"sessions_{email}.csv", mime="text/csv"
                )

with tab_coverage:
    f1, f2, _ = st.columns([1, 1, 2])
    cs_since = f1.date_input("Case studies created since", dt.date(2026, 9, 1))
    att_since = f2.date_input("Attempts started since", dt.date(2026, 9, 15))

    coverage = get_coverage(cs_since, att_since)
    if coverage.empty:
        st.info("No case studies match these filters.")
        st.stop()

    summary = summarise(coverage)

    # Overview across all courses
    by_course = (
        summary.groupby("course_code", as_index=False)
        .agg(attempted=("attempted", "sum"), not_attempted=("not_attempted", "sum"), total_attempts=("total_attempts", "sum"))
        .assign(students=lambda d: d["attempted"] + d["not_attempted"])
        .assign(completion_rate=lambda d: d["attempted"] / d["students"])
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Courses", by_course["course_code"].nunique())
    m2.metric("Students", coverage["email"].nunique())
    m3.metric("Total attempts", int(summary["total_attempts"].sum()))
    m4.metric("Overall completion", f"{summary['attempted'].sum() / summary['students'].sum():.0%}")

    st.subheader("All courses")
    st.caption("Student × case study assignments, stacked by attempt status. Lowest completion at the top.")
    st.altair_chart(coverage_chart(by_course, "course_code", "Course"), width="stretch")

    # Drill down per course
    st.subheader("By course")
    courses = sorted(summary["course_code"].unique())
    course = st.selectbox("Course", courses)
    course_summary = summary[summary["course_code"] == course]

    st.altair_chart(coverage_chart(course_summary, "case_study_title", "Case study"), width="stretch")

    st.dataframe(
        course_summary.drop(columns="course_code").sort_values("completion_rate"),
        width="stretch",
        hide_index=True,
        column_config={
            "case_study_title": "Case study",
            "students": "Enrolled",
            "attempted": "Attempted",
            "not_attempted": "Not attempted",
            "total_attempts": "Total attempts",
            "completion_rate": st.column_config.ProgressColumn("Completion", format="percent", min_value=0, max_value=1),
        },
    )

    with st.expander("Students who have not attempted"):
        case = st.selectbox("Case study", ["All case studies"] + sorted(course_summary["case_study_title"]))
        pending = coverage[(coverage["course_code"] == course) & ~coverage["has_taken"]]
        if case != "All case studies":
            pending = pending[pending["case_study_title"] == case]
        pending = pending[["email", "case_study_title"]].sort_values(["case_study_title", "email"])
        st.write(f"{len(pending)} pending")
        st.dataframe(pending, width="stretch", hide_index=True)
        st.download_button(
            "Download CSV", pending.to_csv(index=False), file_name=f"not_attempted_{course}.csv", mime="text/csv"
        )
