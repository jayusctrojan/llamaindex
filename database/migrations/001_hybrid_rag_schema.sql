-- AI Empire Hybrid RAG Database Schema
-- Version: 2.6.0
-- This creates the minimal SQL tables needed for the hybrid approach

-- =====================================================
-- RECORD MANAGER TABLE
-- =====================================================
-- Tracks document records for hash-based change detection
CREATE TABLE IF NOT EXISTS record_manager_v2 (
    id SERIAL PRIMARY KEY,
    doc_id TEXT UNIQUE NOT NULL,
    hash TEXT NOT NULL,
    graph_id TEXT,
    data_type TEXT CHECK (data_type IN ('unstructured', 'tabular')),
    schema JSONB,
    document_title TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    airtable_record_id TEXT
);

-- Create index for fast lookups
CREATE INDEX IF NOT EXISTS idx_record_manager_doc_id ON record_manager_v2(doc_id);
CREATE INDEX IF NOT EXISTS idx_record_manager_hash ON record_manager_v2(hash);

-- =====================================================
-- SESSION CORRELATION TABLE  
-- =====================================================
-- Manages parallel processing sessions
CREATE TABLE IF NOT EXISTS session_correlation (
    session_id TEXT PRIMARY KEY,
    source_file TEXT NOT NULL,
    processing_paths JSONB NOT NULL, -- {"text": "pending", "image": "completed"}
    correlation_data JSONB, -- Stores extracted data from each path
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'completed', 'failed')),
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- Index for fast session lookups
CREATE INDEX IF NOT EXISTS idx_session_correlation_session_id ON session_correlation(session_id);
CREATE INDEX IF NOT EXISTS idx_session_correlation_status ON session_correlation(status);

-- =====================================================
-- METADATA FIELDS CONFIGURATION
-- =====================================================
-- Dynamic metadata fields for document enrichment
CREATE TABLE IF NOT EXISTS metadata_fields (
    id SERIAL PRIMARY KEY,
    metadata_name TEXT UNIQUE NOT NULL,
    allowed_values TEXT, -- Description of allowed values
    data_type TEXT,
    is_required BOOLEAN DEFAULT FALSE
);

-- Insert default metadata fields
INSERT INTO metadata_fields (metadata_name, allowed_values, data_type, is_required) VALUES
    ('department', 'Engineering, Sales, Marketing, Operations, HR, Finance, Executive', 'text', false),
    ('doc_type', 'Technical, Business, Training, Policy, Report, Presentation', 'text', false),
    ('priority', 'Critical, High, Medium, Low', 'text', false),
    ('confidentiality', 'Public, Internal, Confidential, Restricted', 'text', false),
    ('year', 'Numeric year value', 'number', false),
    ('quarter', 'Q1, Q2, Q3, Q4', 'text', false)
ON CONFLICT (metadata_name) DO NOTHING;

-- =====================================================
-- VECTOR METADATA TABLE
-- =====================================================
-- Fast metadata lookups for vector filtering
CREATE TABLE IF NOT EXISTS vector_metadata (
    id SERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,
    vector_id TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_vector_metadata_doc_id ON vector_metadata(doc_id);
CREATE INDEX IF NOT EXISTS idx_vector_metadata_vector_id ON vector_metadata(vector_id);
CREATE INDEX IF NOT EXISTS idx_vector_metadata_gin ON vector_metadata USING GIN(metadata);

-- =====================================================
-- TABULAR DOCUMENT ROWS TABLE
-- =====================================================
-- Stores tabular data for SQL queries
CREATE TABLE IF NOT EXISTS tabular_document_rows (
    id SERIAL PRIMARY KEY,
    record_manager_id INTEGER REFERENCES record_manager_v2(id) ON DELETE CASCADE,
    row_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for fast queries
CREATE INDEX IF NOT EXISTS idx_tabular_document_rows_record_id ON tabular_document_rows(record_manager_id);
CREATE INDEX IF NOT EXISTS idx_tabular_document_rows_gin ON tabular_document_rows USING GIN(row_data);

-- =====================================================
-- PROCESSING AUDIT LOG TABLE
-- =====================================================
-- Detailed audit trail for all processing operations
CREATE TABLE IF NOT EXISTS processing_audit_log (
    id SERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,
    session_id TEXT,
    action TEXT NOT NULL, -- 'created', 'updated', 'skipped', 'failed'
    processing_type TEXT, -- 'document', 'article', 'video', 'audio'
    hash TEXT,
    chunks_created INTEGER,
    images_extracted INTEGER,
    vision_analyses_performed INTEGER,
    security_scan_result JSONB,
    metadata JSONB,
    error_details TEXT,
    processing_duration_seconds NUMERIC,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for audit queries
CREATE INDEX IF NOT EXISTS idx_processing_audit_doc_id ON processing_audit_log(doc_id);
CREATE INDEX IF NOT EXISTS idx_processing_audit_session_id ON processing_audit_log(session_id);
CREATE INDEX IF NOT EXISTS idx_processing_audit_created_at ON processing_audit_log(created_at);

-- =====================================================
-- HELPER FUNCTIONS
-- =====================================================

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for automatic timestamp updates
DROP TRIGGER IF EXISTS update_record_manager_updated_at ON record_manager_v2;
CREATE TRIGGER update_record_manager_updated_at
    BEFORE UPDATE ON record_manager_v2
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- VIEWS FOR REPORTING
-- =====================================================

-- View for active sessions
CREATE OR REPLACE VIEW active_sessions AS
SELECT 
    session_id,
    source_file,
    processing_paths,
    status,
    created_at,
    EXTRACT(EPOCH FROM (NOW() - created_at)) AS duration_seconds
FROM session_correlation
WHERE status = 'active';

-- View for document processing statistics
CREATE OR REPLACE VIEW document_processing_stats AS
SELECT 
    DATE(created_at) as processing_date,
    COUNT(DISTINCT doc_id) as documents_processed,
    COUNT(CASE WHEN action = 'created' THEN 1 END) as new_documents,
    COUNT(CASE WHEN action = 'updated' THEN 1 END) as updated_documents,
    COUNT(CASE WHEN action = 'skipped' THEN 1 END) as skipped_documents,
    COUNT(CASE WHEN action = 'failed' THEN 1 END) as failed_documents,
    AVG(processing_duration_seconds) as avg_processing_time,
    SUM(chunks_created) as total_chunks_created,
    SUM(images_extracted) as total_images_extracted,
    SUM(vision_analyses_performed) as total_vision_analyses
FROM processing_audit_log
GROUP BY DATE(created_at)
ORDER BY processing_date DESC;

-- =====================================================
-- PERMISSIONS (Adjust based on your Supabase setup)
-- =====================================================

-- Grant permissions to authenticated users
-- Note: Adjust these based on your security requirements

-- Example RLS policies (uncomment and adjust as needed):
-- ALTER TABLE record_manager_v2 ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE session_correlation ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE vector_metadata ENABLE ROW LEVEL SECURITY;

-- CREATE POLICY "Service role can do everything" ON record_manager_v2
--     FOR ALL USING (auth.role() = 'service_role');

-- CREATE POLICY "Authenticated users can read" ON record_manager_v2
--     FOR SELECT USING (auth.role() = 'authenticated');

-- =====================================================
-- MIGRATION NOTES
-- =====================================================
-- 1. Run this script in your Supabase SQL editor
-- 2. Update your environment variables:
--    - SUPABASE_URL
--    - SUPABASE_ANON_KEY
--    - SUPABASE_SERVICE_KEY
-- 3. The hybrid features will automatically use these tables
-- 4. Existing Airtable integration remains for human-readable audit logs
