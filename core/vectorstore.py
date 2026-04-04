import os
import re
from typing import List, Tuple
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from docx import Document
from .models import embeddings
from .text_utils import clean_text
from .chunking import smart_chunking
from .config import DATA_DIR, VECTOR_DIR, QDRANT_API_KEY, QDRANT_URL
from langchain_core.documents import Document as LangChainDocument
import zipfile
import xml.etree.ElementTree as ET
import pickle
import pdfplumber
from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNKS_PICKLE = os.path.join(VECTOR_DIR, "chunks.pkl")
COLLECTION_NAME = "quy_che_db"
SUPPORTED_FORMATS = ('.pdf', '.doc', '.docx')
ACADEMIC_YEAR_PATTERN = re.compile(r"(20\d{2})\s*[-_]\s*(20\d{2})")


def normalize_academic_year(start_year: str, end_year: str) -> str:
    return f"{int(start_year):04d}-{int(end_year):04d}"


def extract_academic_year(text: str) -> str:
    if not text:
        return ""
    match = ACADEMIC_YEAR_PATTERN.search(text)
    if not match:
        return ""
    return normalize_academic_year(match.group(1), match.group(2))


def discover_data_files() -> List[Tuple[str, str, str, str]]:
    """Quet de quy thu muc data va tra ve (filepath, filename, relpath, academic_year)."""
    if not os.path.isdir(DATA_DIR):
        return []

    discovered = []
    for root, _, files in os.walk(DATA_DIR):
        for filename in files:
            if not filename.lower().endswith(SUPPORTED_FORMATS):
                continue

            filepath = os.path.join(root, filename)
            relpath = os.path.relpath(filepath, DATA_DIR)
            year = extract_academic_year(relpath) or "ALL"
            discovered.append((filepath, filename, relpath, year))

    discovered.sort(key=lambda x: x[2].lower())
    return discovered


def collect_chunk_relpaths(chunks: List) -> set:
    relpaths = set()
    for chunk in chunks:
        metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        relpath = metadata.get("source_relpath")
        if relpath:
            relpaths.add(os.path.normpath(str(relpath)))
    return relpaths


def enrich_chunk_metadata(chunks: List) -> bool:
    """Bo sung metadata nam hoc cho chunks cu de dam bao loc theo nam hoat dong."""
    changed = False
    for chunk in chunks:
        metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}

        source = metadata.get("source")
        source_file = metadata.get("source_file")
        source_relpath = metadata.get("source_relpath")

        if not source_relpath:
            if source and os.path.isabs(str(source)):
                try:
                    source_relpath = os.path.relpath(source, DATA_DIR)
                except Exception:
                    source_relpath = str(source)
            elif source:
                source_relpath = str(source)
            elif source_file:
                source_relpath = str(source_file)

            if source_relpath:
                metadata["source_relpath"] = source_relpath
                changed = True

        if not metadata.get("academic_year"):
            year = extract_academic_year(source_relpath or source_file or "") or "ALL"
            metadata["academic_year"] = year
            changed = True

        if "page_number" not in metadata and metadata.get("page") is not None:
            metadata["page_number"] = metadata.get("page")
            changed = True

        chunk.metadata = metadata

    return changed


def load_and_clean_all_docs() -> List[LangChainDocument]:
    docs: List[LangChainDocument] = []
    file_entries = discover_data_files()

    if not file_entries:
        logger.error(" Không tìm thấy file PDF, DOC, hoặc DOCX!")
        return docs

    for filepath, filename, relpath, academic_year in file_entries:
        logger.info(f" Đang đọc: {relpath}")
        loaded_docs = load_documents_from_file(filepath, filename)

        for i, doc in enumerate(loaded_docs, 1):
            cleaned = clean_text(doc.page_content)
            if not cleaned or len(cleaned.split()) < 20:
                continue

            page_number = doc.metadata.get("page") if isinstance(doc.metadata, dict) else None
            if page_number is None:
                page_number = i

            doc.metadata["source_file"] = filename
            doc.metadata["source_relpath"] = relpath
            doc.metadata["academic_year"] = academic_year
            doc.metadata["page_number"] = page_number
            doc.page_content = cleaned
            docs.append(doc)

    return docs

def table_to_markdown(data: List[List[str]]) -> str:
    if not data or len(data) < 2:
        return ""
    header = data[0]
    header = [str(cell).replace('\n', ' ').strip() if cell else "" for cell in header]
    separator = ["---"] * len(header)
    markdown_lines = []
    markdown_lines.append("| " + " | ".join(header) + " |")
    markdown_lines.append("| " + " | ".join(separator) + " |")
    for row in data[1:]:
        clean_row = [str(cell).replace('\n', '<br>').strip() if cell else "" for cell in row]
        markdown_lines.append("| " + " | ".join(clean_row) + " |")
    return "\n".join(markdown_lines) + "\n\n"

def read_pdf_with_tables(filepath: str) -> List[LangChainDocument]:
    docs = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                tables = page.extract_tables()
                table_texts = []
                if tables:
                    for table in tables:
                        md_table = table_to_markdown(table)
                        if md_table:
                            table_texts.append(md_table)
                full_content = text + "\n\n[BẢNG DỮ LIỆU TRÍCH XUẤT]:\n" + "\n".join(table_texts)
                if full_content.strip():
                    docs.append(LangChainDocument(
                        page_content=full_content,
                        metadata={"source": filepath, "page": i}
                    ))
    except Exception as e:
        logger.error(f"Lỗi đọc PDF (pdfplumber) {os.path.basename(filepath)}: {e}")
    return docs

def iter_block_items(parent):
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("Chỉ hỗ trợ duyệt Document hoặc Cell")
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)

def read_docx_with_tables(filepath: str) -> str:
    doc = Document(filepath)
    full_text = []
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            if block.text.strip():
                full_text.append(block.text.strip())
        elif isinstance(block, Table):
            table_data = []
            for row in block.rows:
                row_data = []
                for cell in row.cells:
                    cell_text = clean_text(cell.text)
                    row_data.append(cell_text)
                table_data.append(row_data)
            md_table = table_to_markdown(table_data)
            if md_table:
                full_text.append(f"\n{md_table}\n")
    return "\n".join(full_text)

def extract_text_from_doc_com(filepath: str) -> str:
    try:
        import win32com.client
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(os.path.abspath(filepath))
        text = doc.Range().Text
        doc.Close()
        word.Quit()
        return text.strip()
    except Exception as e:
        logger.error(f" COM API lỗi: {str(e)[:40]}")
        return ""

def extract_text_from_doc(filepath: str) -> str:
    try:
        doc = Document(filepath)
        text = "\n".join([para.text for para in doc.paragraphs])
        if text.strip():
            return text
    except Exception as e:
        logger.error(f"Lỗi đọc DOC {os.path.basename(filepath)}: {e}")
    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            xml_content = zip_ref.read('word/document.xml')
            root = ET.fromstring(xml_content)
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            paragraphs = root.findall('.//w:p', ns)
            text_list = []
            for para in paragraphs:
                texts = para.findall('.//w:t', ns)
                para_text = ''.join([t.text for t in texts if t.text])
                if para_text.strip():
                    text_list.append(para_text)
            return "\n".join(text_list)
    except Exception as e:
        logger.error(f" Lỗi đọc DOCX {os.path.basename(filepath)}: {e}")
        pass
    if filepath.lower().endswith('.doc'):
        return extract_text_from_doc_com(filepath)
    return ""

def load_doc_file(filepath: str) -> List[LangChainDocument]:
    docs = []
    try:
        text = extract_text_from_doc(filepath)
        if text.strip():
            docs.append(LangChainDocument(page_content=text, metadata={"source": filepath}))
        else:
            logger.warning(f" File rỗng: {os.path.basename(filepath)}")
    except Exception as e:
        logger.error(f" Không thể đọc {os.path.basename(filepath)}: {str(e)[:60]}")
    return docs

def load_documents_from_file(filepath: str, filename: str) -> List:
    docs = []
    try:
        if filename.lower().endswith('.pdf'):
            docs = read_pdf_with_tables(filepath)
        elif filename.lower().endswith('.docx'):
            text = read_docx_with_tables(filepath)
            if text:
                docs = [LangChainDocument(page_content=text, metadata={"source": filepath})]
        elif filename.lower().endswith('.doc'):
            docs = load_doc_file(filepath)
        
        if docs:
            logger.info(f" Đã đọc: {filename}")
        return docs
    except Exception as e:
        logger.error(f" Lỗi đọc {filename}: {str(e)[:60]}")
        return []

def build_vectorstore_improved(recreate_collection: bool = False) -> Tuple[QdrantVectorStore, List]:
    logger.info(" Đang xây dựng vectorstore...")
    docs = load_and_clean_all_docs()
    
    if not docs:
        logger.error(" Không có văn bản hợp lệ!")
        return None, []
    
    logger.info(f" Đã đọc {len(docs)} trang hợp lệ")
    chunks = smart_chunking(docs)
    logger.info ("Đang kết nối với và đẩy dữ liệu lên Qdrant Cloud ")

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    
    if client.collection_exists(COLLECTION_NAME):
        if recreate_collection:
            logger.warning(f"Collection {COLLECTION_NAME} đã tồn tại. Đang tạo lại để đồng bộ dữ liệu mới...")
            client.delete_collection(collection_name=COLLECTION_NAME)
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
    else:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

    db = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    #Đẩy chunks lên cloud 
    db.add_documents(chunks)

    #Lưu chunk local 
    try:
        os.makedirs(VECTOR_DIR, exist_ok=True)
        with open(CHUNKS_PICKLE, 'wb') as f:
            pickle.dump(chunks, f)
        logger.info(f" Đã lưu chunks vào {CHUNKS_PICKLE}")
    except Exception as e:
        logger.error(f" Không thể lưu chunks: {e}")

    logger.info(" Hoàn tất xây dựng và đưa lên Qdrant Cloud")
    return db, chunks

def load_vectorstore_improved() -> Tuple[QdrantVectorStore, List]:
    logger.info("Đang tải vectorstore  từ Qdrant Cloud")
    
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    db = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    # Load chunks từ file pickle nếu có, để tránh phải tái tạo từ file nguồn mỗi lần khởi động
    if os.path.exists(CHUNKS_PICKLE):
        try:
            with open(CHUNKS_PICKLE, 'rb') as f:
                chunks = pickle.load(f)
            if enrich_chunk_metadata(chunks):
                try:
                    os.makedirs(VECTOR_DIR, exist_ok=True)
                    with open(CHUNKS_PICKLE, 'wb') as f:
                        pickle.dump(chunks, f)
                    logger.info(" Đã cập nhật metadata năm học cho chunks local")
                except Exception as e:
                    logger.error(f" Không thể cập nhật {CHUNKS_PICKLE}: {e}")

            discovered_relpaths = {os.path.normpath(relpath) for _, _, relpath, _ in discover_data_files()}
            chunk_relpaths = collect_chunk_relpaths(chunks)
            missing_relpaths = sorted(discovered_relpaths - chunk_relpaths)

            if missing_relpaths:
                logger.warning(
                    f" Phát hiện {len(missing_relpaths)} file mới chưa có trong chunks cache. Đang build lại vectorstore theo dữ liệu hiện tại..."
                )
                return build_vectorstore_improved(recreate_collection=True)

            logger.info(f" Đã load {len(chunks)} chunks từ {CHUNKS_PICKLE}")
            return db, chunks
        except Exception as e:
            logger.error(f" Không thể đọc {CHUNKS_PICKLE}: {e} — sẽ thử tái tạo từ file nguồn.")


    # Nếu mất file pickle hoặc lỗi, fallback về tái tạo từ file nguồn 
    docs = load_and_clean_all_docs()

    chunks = smart_chunking(docs)
    # Lưu lại chunks mới tái tạo vào file pickle để lần sau load nhanh hơn
    try:
        os.makedirs(VECTOR_DIR, exist_ok=True)
        with open(CHUNKS_PICKLE, 'wb') as f:
            pickle.dump(chunks, f)
        logger.info(f" Đã tái tạo và lưu {len(chunks)} chunks vào {CHUNKS_PICKLE}")
    except Exception as e:
        logger.error(f" Không thể lưu chunks: {e}")

    logger.info(f"Đã tái tạo {len(chunks)} chunks từ file nguồn")
    return db, chunks