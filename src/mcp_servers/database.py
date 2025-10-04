"""
authors: Emile Johnston, Piotr Fic (adapted for SQL Server on Azure)
date: 2025-10-02

MCP server that connects directly to a remote SQL Server database
"""

import os
import re
import pymssql
from typing import Any
from mcp.server.fastmcp import FastMCP
from contextlib import contextmanager
from dotenv import load_dotenv

mcp = FastMCP("Remote_SQLServer")

load_dotenv()

@contextmanager
def get_conn():
    conn = pymssql.connect(
        server=os.getenv("SQLSERVER_HOST"),
        port=int(os.getenv("SQLSERVER_PORT", 1433)),
        user=os.getenv("SQLSERVER_USER"),
        password=os.getenv("SQLSERVER_PASSWORD"),
        database=os.getenv("SQLSERVER_DB"),
        timeout=30,
        login_timeout=30,
        as_dict=False,
    )
    try:
        yield conn
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _rows_to_lists(rows):
    return [list(r) for r in rows] if rows else []


_identifier_re = re.compile(r"^[A-Za-z0-9_$]+(\.[A-Za-z0-9_$]+)?$")


def _quote_ident(name: str) -> str:
    return f"[{name.replace(']', ']]')}]"


def _qualify(table_name: str) -> str:
    """Return a safely quoted identifier. Supports `db.table` or `table`."""
    if not _identifier_re.match(table_name):
        raise ValueError("Invalid table name")
    parts = table_name.split(".")
    if len(parts) == 2:
        return f"{_quote_ident(parts[0])}.{_quote_ident(parts[1])}"
    return _quote_ident(parts[0])


# -----------------------------------------------------------------------------
# Tools / Resources (SAME interface as your original code)
# -----------------------------------------------------------------------------


@mcp.tool()
def query_database(sql_query: str) -> list[list[Any]]:
    """
    Execute a SQL query on the remote SQL Server database.
    Returns SELECT rows; for non-SELECT, returns [["rows_affected", <int>]].
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query)
                try:
                    rows = cur.fetchall()
                    return _rows_to_lists(rows)
                except pymssql.ProgrammingError:
                    affected = cur.rowcount
                    conn.commit()
                    return [["rows_affected", affected]]
    except Exception as e:
        return [["Error", str(e)]]


@mcp.resource("schema://{table_name}")
def get_schema(table_name: str) -> str:
    """
    Get the CREATE TABLE DDL of a specific table in the remote SQL Server database.
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT 
                        COLUMN_NAME,
                        DATA_TYPE,
                        IS_NULLABLE,
                        CHARACTER_MAXIMUM_LENGTH,
                        COLUMN_DEFAULT
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME = '{table_name.split(".")[-1]}'
                    ORDER BY ORDINAL_POSITION
                """)
                rows = cur.fetchall()
                if not rows:
                    return "No schema found."

                # Build a simple schema description
                schema_parts = [f"Table: {table_name}"]
                for row in rows:
                    col_name, data_type, nullable, max_length, default = row
                    nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
                    length_str = f"({max_length})" if max_length else ""
                    default_str = f" DEFAULT {default}" if default else ""
                    schema_parts.append(
                        f"  {col_name} {data_type}{length_str} {nullable_str}{default_str}"
                    )

                return "\n".join(schema_parts)
    except Exception as e:
        return f"Error: {e}"


@mcp.resource("tables://")
def get_tables() -> list[str]:
    """
    Get the list of tables in the remote SQL Server database.
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
                )
                return [row[0] for row in cur.fetchall()]
    except Exception as e:
        return [f"Error: {e}"]


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")