"""Generate domain-specific test datasets for Corporate Finance, M&A, and Venture Capital.

Creates both .txt and high-quality standard .pdf documents in data/raw/.
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

DATA_RAW = Path("data/raw")
DATA_RAW.mkdir(parents=True, exist_ok=True)

# ── Dataset 1: Tech Venture & M&A Report ───────────────────────────────────────
DOC1_TITLE = "Tech Venture Capital and M&A Deal Report"
DOC1_PARAGRAPHS = [
    "Microsoft invested in OpenAI to expand enterprise artificial intelligence infrastructure. "
    "Satya Nadella serves as CEO of Microsoft in Redmond, Washington. "
    "Sam Altman co-founded OpenAI with Greg Brockman in San Francisco, California. "
    "OpenAI established a partnership with Apple to integrate conversational intelligence into consumer devices. "
    "Tim Cook leads Apple in Cupertino, California. "
    "Thrive Capital invested in OpenAI during its latest valuation round.",

    "Patrick Collison co-founded Stripe in San Francisco. "
    "Sequoia Capital backed Stripe across multiple venture capital rounds. "
    "Roelof Botha leads Sequoia Capital in Menlo Park, California. "
    "Stripe acquired Bridge to accelerate global payment settlements. "
    "Peter Thiel invested in Stripe during early financing rounds. "
    "Peter Thiel co-founded Founders Fund in San Francisco.",

    "Microsoft acquired GitHub for $7.5 billion in an all-stock transaction. "
    "Databricks acquired MosaicML to strengthen generative model pre-training. "
    "Andreessen Horowitz invested in Databricks alongside Tiger Global. "
    "Marc Andreessen co-founded Andreessen Horowitz in Silicon Valley. "
    "Benchmark backed Uber in its initial funding stages. "
    "Dara Khosrowshahi serves as CEO of Uber in San Francisco.",
]

# ── Dataset 2: Private Equity, Buyouts & Banking ──────────────────────────────
DOC2_TITLE = "Private Equity Buyouts and Corporate Restructuring Brief"
DOC2_PARAGRAPHS = [
    "KKR acquired BMC Software in Houston, Texas to expand automated enterprise software operations. "
    "Blackstone invested in Databricks to accelerate cloud data analytics expansion. "
    "Apollo Global Management acquired Yahoo for $5 billion in a corporate carve-out transaction.",

    "Goldman Sachs advised Microsoft on its acquisition of GitHub in San Francisco. "
    "Morgan Stanley financed Stripe in a major capital liquidity facility. "
    "SoftBank backed OpenAI through structured convertible investments in Silicon Valley.",

    "Berkshire Hathaway holds a significant investment in Apple. "
    "Warren Buffett leads Berkshire Hathaway in Omaha, Nebraska. "
    "Salesforce acquired Slack to integrate real-time collaboration into cloud software. "
    "Marc Benioff leads Salesforce in San Francisco, California.",
]


def create_txt(filename: str, paragraphs: list[str]) -> Path:
    path = DATA_RAW / filename
    text = "\n\n".join(paragraphs)
    path.write_text(text, encoding="utf-8")
    print(f"Created {path}")
    return path


def create_pdf(filename: str, title: str, paragraphs: list[str]) -> Path:
    path = DATA_RAW / filename
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=26,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=14,
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#334155"),
        spaceAfter=12,
    )

    elements = [
        Paragraph(title, title_style),
        Spacer(1, 10),
    ]

    for p in paragraphs:
        elements.append(Paragraph(p, body_style))
        elements.append(Spacer(1, 6))

    doc.build(elements)
    print(f"Created {path}")
    return path


def main():
    # Dataset 1
    create_txt("tech_venture_and_ma_report.txt", DOC1_PARAGRAPHS)
    create_pdf("tech_venture_and_ma_report.pdf", DOC1_TITLE, DOC1_PARAGRAPHS)

    # Dataset 2
    create_txt("private_equity_and_buyouts.txt", DOC2_PARAGRAPHS)
    create_pdf("private_equity_and_buyouts.pdf", DOC2_TITLE, DOC2_PARAGRAPHS)


if __name__ == "__main__":
    main()

