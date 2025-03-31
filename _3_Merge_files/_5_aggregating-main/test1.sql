-- 기본 테이블
CREATE TABLE sentiment_indicators (
    stock_code VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    hour TINYINT NOT NULL,
    fear_ratio DECIMAL(5, 4) NOT NULL,
    neutral_ratio DECIMAL(5, 4) NOT NULL,
    greed_ratio DECIMAL(5, 4) NOT NULL,
    fear_greed_index DECIMAL(5, 2) NOT NULL, 
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_code, date, hour)
);

-- 월별 평균 테이블
CREATE TABLE monthly_feargreed (
    stock_code VARCHAR(10) NOT NULL,
    year SMALLINT NOT NULL,
    month TINYINT NOT NULL,
    avg_fear_greed DECIMAL(5, 2) NOT NULL, 
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_code, year, month)
);
