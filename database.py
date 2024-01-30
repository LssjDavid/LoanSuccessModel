import sqlite3
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Creating an empty database
Path("bank.db").touch()

# Connect to database
conn = sqlite3.connect("bank.db")
c = conn.cursor()

# Creating a table
c.execute(
    """CREATE TABLE bank (
    age int, job text,
    marital text, education text,
    default_e text, balance int,
    housing text, loan text,
    contact text, day int,
    month text, duration int,
    campaign int, pdays text,
    previous int, poutcome text,
    deposit text
    );"""
)






