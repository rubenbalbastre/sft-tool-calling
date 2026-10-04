"""Small SQLite master-data store for the procurement environment."""

import json
import sqlite3
from pathlib import Path


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[3] / "data/environment/procurement.db"

MATERIALS = [
    ("MAT-1042", "Stainless steel sheet", "kg"),
    ("MAT-2031", "Industrial resin", "kg"),
    ("MAT-3305", "Precision bearings", "units"),
]

SUPPLIERS = [
    ("SUP-001", "Iberian Metals", "Spain", 0.97, "low", ["ISO-9001", "ISO-14001"]),
    ("SUP-002", "Catalonia Industrial", "Spain", 0.95, "medium", ["ISO-9001"]),
    ("SUP-003", "Metaux du Sud", "France", 0.96, "low", ["ISO-9001", "ISO-14001"]),
    ("SUP-004", "Rhone Components", "France", 0.93, "medium", ["ISO-9001"]),
    ("SUP-005", "Rhein Werkstoffe", "Germany", 0.98, "low", ["ISO-9001", "ISO-14001"]),
    ("SUP-006", "Bavaria Components", "Germany", 0.94, "medium", ["ISO-9001"]),
    ("SUP-007", "Lombardia Supply", "Italy", 0.95, "low", ["ISO-9001", "ISO-14001"]),
    ("SUP-008", "Torino Industrial", "Italy", 0.92, "high", ["ISO-9001"]),
]

SUPPLIER_MATERIALS = [
    ("SUP-001", "MAT-1042", 5.50, 2200, 4),
    ("SUP-001", "MAT-2031", 3.80, 1800, 3),
    ("SUP-002", "MAT-1042", 5.10, 1700, 6),
    ("SUP-002", "MAT-3305", 12.00, 2400, 5),
    ("SUP-003", "MAT-1042", 5.35, 2000, 5),
    ("SUP-003", "MAT-2031", 3.65, 2100, 4),
    ("SUP-004", "MAT-2031", 3.40, 1600, 7),
    ("SUP-004", "MAT-3305", 11.50, 1800, 6),
    ("SUP-005", "MAT-1042", 5.80, 2600, 3),
    ("SUP-005", "MAT-3305", 12.40, 3000, 3),
    ("SUP-006", "MAT-2031", 3.55, 2300, 5),
    ("SUP-006", "MAT-3305", 11.80, 2200, 5),
    ("SUP-007", "MAT-1042", 5.25, 1900, 5),
    ("SUP-007", "MAT-2031", 3.70, 2500, 4),
    ("SUP-008", "MAT-1042", 4.95, 1500, 7),
    ("SUP-008", "MAT-3305", 10.90, 2000, 6),
]


def build_database(path=DEFAULT_DATABASE_PATH, seed=42):
    """Create the fixed master-data database if it does not already exist."""
    path = Path(path)
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE materials (
                material_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                unit TEXT NOT NULL
            );
            CREATE TABLE suppliers (
                supplier_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                country TEXT NOT NULL,
                reliability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                certifications TEXT NOT NULL
            );
            CREATE TABLE supplier_materials (
                supplier_id TEXT NOT NULL,
                material_id TEXT NOT NULL,
                base_unit_price REAL NOT NULL,
                normal_capacity REAL NOT NULL,
                preparation_days INTEGER NOT NULL,
                PRIMARY KEY (supplier_id, material_id),
                FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id),
                FOREIGN KEY (material_id) REFERENCES materials(material_id)
            );
            """
        )
        connection.executemany("INSERT INTO materials VALUES (?, ?, ?)", MATERIALS)
        connection.executemany(
            "INSERT INTO suppliers VALUES (?, ?, ?, ?, ?, ?)",
            [(*row[:5], json.dumps(row[5])) for row in SUPPLIERS],
        )
        connection.executemany(
            "INSERT INTO supplier_materials VALUES (?, ?, ?, ?, ?)",
            SUPPLIER_MATERIALS,
        )
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [("schema_version", "1"), ("database_seed", str(seed))],
        )
        connection.commit()
    finally:
        connection.close()
    return path


class ProcurementRepository:
    """Read-only access to fixed procurement master data."""

    def __init__(self, path=DEFAULT_DATABASE_PATH):
        self.path = build_database(path)
        self.connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row

    def material(self, material_id):
        row = self.connection.execute(
            "SELECT * FROM materials WHERE material_id = ?", (material_id,)
        ).fetchone()
        return dict(row) if row else None

    def search_suppliers(self, material_id, countries=None):
        rows = self.connection.execute(
            """
            SELECT s.supplier_id, s.name, s.country
            FROM suppliers s
            JOIN supplier_materials sm USING (supplier_id)
            WHERE sm.material_id = ?
            ORDER BY s.supplier_id
            """,
            (material_id,),
        ).fetchall()
        allowed = set(countries or [])
        return [dict(row) for row in rows if not allowed or row["country"] in allowed]

    def supplier(self, supplier_id):
        row = self.connection.execute(
            "SELECT * FROM suppliers WHERE supplier_id = ?", (supplier_id,)
        ).fetchone()
        if not row:
            return None
        supplier = dict(row)
        supplier["certifications"] = json.loads(supplier["certifications"])
        return supplier

    def supplier_material(self, supplier_id, material_id):
        row = self.connection.execute(
            """
            SELECT sm.*, m.unit
            FROM supplier_materials sm
            JOIN materials m USING (material_id)
            WHERE sm.supplier_id = ? AND sm.material_id = ?
            """,
            (supplier_id, material_id),
        ).fetchone()
        return dict(row) if row else None

    def close(self):
        self.connection.close()
