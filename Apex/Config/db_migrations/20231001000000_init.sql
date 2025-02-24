CREATE TABLE IF NOT EXISTS assets (
    symbol TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    exchange TEXT NOT NULL,
    price REAL DEFAULT 1.0,
    volume INTEGER NOT NULL CHECK (volume > 0)
);
