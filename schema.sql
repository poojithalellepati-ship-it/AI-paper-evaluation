-- ============================================================
-- AI Answer Evaluation System - MySQL Schema
-- ============================================================

CREATE DATABASE IF NOT EXISTS ai_evaluation_db;
USE ai_evaluation_db;

-- Students Table
CREATE TABLE IF NOT EXISTS students (
    student_id VARCHAR(20) PRIMARY KEY,
    student_name VARCHAR(100) NOT NULL,
    subject_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Uploaded Files Table
CREATE TABLE IF NOT EXISTS uploaded_files (
    file_id INT AUTO_INCREMENT PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    file_type ENUM('question_paper', 'key_answer', 'student_answer') NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    student_id VARCHAR(20),
    subject_name VARCHAR(100),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE SET NULL
);

-- Evaluation Sessions Table
CREATE TABLE IF NOT EXISTS evaluation_sessions (
    session_id INT AUTO_INCREMENT PRIMARY KEY,
    session_name VARCHAR(100),
    total_students INT DEFAULT 0,
    evaluated_students INT DEFAULT 0,
    avg_score FLOAT DEFAULT 0,
    pass_percentage FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Evaluation Results Table
CREATE TABLE IF NOT EXISTS evaluation_results (
    evaluation_id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT,
    student_id VARCHAR(20) NOT NULL,
    question_number INT NOT NULL,
    question_text TEXT,
    key_answer TEXT,
    student_answer TEXT,
    similarity_score FLOAT DEFAULT 0,
    confidence_score FLOAT DEFAULT 0,
    obtained_marks FLOAT DEFAULT 0,
    total_marks FLOAT DEFAULT 5,
    ai_remarks TEXT,
    keyword_match_count INT DEFAULT 0,
    evaluation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES evaluation_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
);

-- Student Final Results Table
CREATE TABLE IF NOT EXISTS student_final_results (
    result_id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT,
    student_id VARCHAR(20) NOT NULL,
    student_name VARCHAR(100),
    subject_name VARCHAR(100),
    total_obtained FLOAT DEFAULT 0,
    total_marks FLOAT DEFAULT 0,
    percentage FLOAT DEFAULT 0,
    grade VARCHAR(5),
    pass_fail ENUM('Pass', 'Fail') DEFAULT 'Fail',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES evaluation_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
);

-- Analytics Logs Table
CREATE TABLE IF NOT EXISTS analytics_logs (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    action_type VARCHAR(100),
    details TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sample Data Inserts
INSERT INTO evaluation_sessions (session_name, total_students, evaluated_students, avg_score, pass_percentage)
VALUES ('Demo Session 2025', 3, 3, 72.5, 66.7);

INSERT INTO students VALUES
    ('STU001', 'Arjun Sharma', 'Computer Science'),
    ('STU002', 'Priya Patel', 'Data Structures'),
    ('STU003', 'Ravi Kumar', 'Machine Learning');
