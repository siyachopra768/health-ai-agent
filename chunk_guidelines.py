import json
import logging
from pathlib import Path
from typing import List,Dict,Any,Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from pypdf import PdfReader
from bs4 import BeautifulSoup

DATA_DIR = Path("data/guidelines")
CHUNK_DIR = Path("data/chunks")
CHUNK_DIR.mkdir(parents=True,exist_ok=True)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"


)
logger = logging.getLogger(__name__)

class GuidelineChunker:
    def __init__(
    self,
    data_dir: Path = DATA_DIR,
    chunk_dir: Path = CHUNK_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: Optional[List[str]] = None,
):
     self.data_dir = data_dir
     self.chunk_dir = chunk_dir
     self.chunk_dir.mkdir(parents=True, exist_ok=True)
     self.chunk_size = chunk_size
     self.chunk_overlap = chunk_overlap
     self.splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or CHUNK_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    ) 


    def _extract_pdf(self,filepath: Path)-> str:
        reader = PdfReader(str(filepath))
        pages = ((i, (page.extract_text() or "").strip()) for i, page in enumerate(reader.pages))
        return "\n\n".join(f"[Page {i+1}]\n{text}" for i, text in pages if text)
            
         

    def _extract_html(self,filepath: Path) -> str:
        with open(filepath,encoding="utf-8",errors="ignore") as f:
            soup = BeautifulSoup(f.read(),"html.parser")
        for tag in soup(["script","style","nav","footer","header","aside"]):
            tag.decompose()

        return "\n".join(line.strip() for line in soup.get_text("\n").split("\n") if line.strip())


    def _extract_text(self,filepath: Path)-> str:
        extractors = {
            ".pdf": self._extract_pdf,
            ".html": self._extract_html,
            ".htm": self._extract_html,
            ".txt": lambda p: p.read_text(encoding="utf-8",errors="ignore"),

        }
        extractor = extractors.get(filepath.suffix.lower())
        if not extractor:
             raise ValueError(f"Unsupported file type: {filepath.suffix}")
        return extractor(filepath)

    def _load_metadata(self,guideline_id: str)-> Dict[str,Any]:
        meta_path = self.data_dir / f"{guideline_id}_metadata.json"
        return json.loads(meta_path.read_text())

    def _create_chunks(self,text: str,metadata:Dict[str,Any]) -> List[Document]:
        doc = Document(page_content=text,metadata=metadata)
        chunks = self.splitter.split_documents([doc])
        for i,chunk in enumerate(chunks):
            chunk.metadata.update(
                chunk_index=i,
                total_chunks=len(chunks),
                chunk_size=len(chunk.page_content),
                chunk_id=f"{metadata['id']}_chunk_{i:04d}",
            )
        return chunks

    def _save_chunks(self,chunks: List[Document],guideline_id: str)-> Path:
        output = self.chunk_dir / f"{guideline_id}_chunks.json"
        output.write_text(
            json.dumps(
                [
                    {"chunk_id": c.metadata["chunk_id"], "content": c.page_content, "metadata": c.metadata}
                    for c in chunks
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return output

    def process(self, guideline_id: str) -> Dict[str, Any]:
        """Process a single guideline end-to-end."""
        logger.info("Processing: %s", guideline_id)

        # Find source file
        source_files = list(self.data_dir.glob(f"{guideline_id}.*"))
        source = next((f for f in source_files if f.suffix in {".pdf", ".html", ".htm", ".txt"}), None)
        if not source:
            return {"guideline_id": guideline_id, "status": "error", "error": "No source file found"}

        # Load metadata
        try:
            metadata = self._load_metadata(guideline_id)
        except FileNotFoundError:
            return {"guideline_id": guideline_id, "status": "error", "error": "No metadata file found"}

        # Extract & chunk
        try:
            text = self._extract_text(source)
            logger.info("  Extracted %d characters from %s", len(text), source.name)
        except Exception as e:
            return {"guideline_id": guideline_id, "status": "error", "error": f"Extraction failed: {e}"}
        if not text.strip():
             return {"guideline_id": guideline_id, "status": "error", "error": "Empty text extracted"}

        chunks = self._create_chunks(text, metadata)
        logger.info("  Created %d chunks", len(chunks))

        output_path = self._save_chunks(chunks, guideline_id)
        logger.info("  Saved to %s", output_path)

        return {
            "guideline_id": guideline_id,
            "status": "success",
            "source_file": str(source),
            "text_length": len(text),
            "num_chunks": len(chunks),
            "output_file": str(output_path),
        }

    def process_all(self) -> Dict[str, Any]:
        """Process all guidelines from manifest."""
        manifest_path = self.data_dir / "collection_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("Run collect_guidelines.py first - no manifest found")

        guidelines = [g["id"] for g in json.loads(manifest_path.read_text()).get("guidelines", [])]
        logger.info("Found %d guidelines to process", len(guidelines))

        results = [self.process(gid) for gid in guidelines]

        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]

        manifest = {
            "total_guidelines": len(guidelines),
            "successful": len(successful),
            "failed": len(failed),
            "total_chunks": sum(r.get("num_chunks", 0) for r in successful),
            "total_characters": sum(r.get("text_length", 0) for r in successful),
            "chunk_config": {"chunk_size": self.splitter._chunk_size, "chunk_overlap": self.splitter._chunk_overlap},
            "results": results,
        }

        (self.chunk_dir / "chunking_manifest.json").write_text(json.dumps(manifest, indent=2))

        logger.info("=" * 50)
        logger.info("CHUNKING COMPLETE: %d/%d successful, %d total chunks",
                    len(successful), len(guidelines), manifest["total_chunks"])
        if failed:
             for r in failed:
                logger.error("  Failed: %s - %s", r["guideline_id"], r["error"])

        return manifest


def main():
    chunker = GuidelineChunker()
    chunker.process_all()


if __name__ == "__main__":
    main()

        
        