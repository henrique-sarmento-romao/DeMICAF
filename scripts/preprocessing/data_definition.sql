PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS User(
    username TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS Dataset(
    name TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS Image(
    id INTEGER PRIMARY KEY,
    path TEXT,
    dataset TEXT REFERENCES Dataset
);

CREATE TABLE IF NOT EXISTS Compliancy(
    compliancy TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS Annotation(
    image INTEGER REFERENCES Image,
    annotator TEXT REFERENCES User,
    compliancy TEXT REFERENCES Compliancy,
    PRIMARY KEY (image, annotator)
);

CREATE TABLE IF NOT EXISTS Cause(
    id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS AnnotationCause(
    image INTEGER,
    annotator TEXT,
    cause INTEGER REFERENCES Cause,
    PRIMARY KEY (image, annotator, cause),
    FOREIGN KEY (image, annotator) REFERENCES Annotation(image, annotator)
);