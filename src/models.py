from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy import Column, Integer, Float, Boolean, String, Interval, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, backref, sessionmaker
from sqlalchemy_utils import database_exists, create_database
import psycopg2
from src.db_settings import settings
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy import PickleType


def get_engine(user, passwrd, host, port, db):

    db_string = f"postgresql://{user}:{passwrd}@{host}:{port}/{db}"
    #db_string = 'postgresql://myuser:mypassword@db/causal_db'
    if not database_exists:
        create_database(db_string)
    engine = create_engine(db_string)
    return engine

engine = get_engine(settings['user'],
                    settings['passwrd'],
                    settings['host'],
                    settings['port'],
                    settings['db'])
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
connection = engine.connect()

Base = declarative_base()


class Datasets(Base):
    __tablename__ = 'DATASETS'
    id = Column(Integer(), primary_key=True, index=True, unique=True)
    dataset_link = Column(String(), nullable=False)
    rows_number = Column(Integer())
    columns_number = Column(Integer())
    baseline = Column(Float())
    best_run = Column(String())
    runs = relationship('Runs', backref='dataset')


class Runs(Base):
    __tablename__ = 'RUNS'

    id = Column(Integer(), primary_key=True, index=True, unique=True)
    run_id = Column(String())
    dataset_id = Column(Integer(), ForeignKey('DATASETS.id'))
    best_result = Column(Float())
    data_type = Column(String())
    best_algorithm = Column(String())
    best_test = Column(String())
    best_alpha = Column(Integer())
    alphas = Column(MutableList.as_mutable(PickleType), default=[])
    methods = Column(MutableList.as_mutable(PickleType),default=[])
    uc_priority = Column(Integer())
    uc_rule = Column(Integer())
    kernelX = Column(String())
    kernelY = Column(String())
    polyd = Column(Integer())
    kwidthx = Column(Float())
    kwidthy = Column(Float())


Base.metadata.create_all(bind=engine)
