# Dataset-Database

The Dataset-Database is an entity that allows you for annotating your whole dataset. This is intended to allow you for a couple of useful things. These things include but are not limited to: Rejecting files and preventing them from entering training, filtering files w.r.t. KPIs, sorting files w.r.t. KPIs.

## How to use CLI

The CLI as it name suggests is supposed to be executed from the command-line. It allows you for quick-interaction with the database. Note that the Python-API is an alternative that might have more power in specific cases.

### Create an initial, empty database

This needs to be only when you initialize the whole setup. Creates some tables but does not provide content.

```bash
python dsdb_cli.py init
```

### Update existing database with ETL data

Goes through the whole ETL data and updates the database. Maintains existing entries. Inserts new ones. This might take a while. Especially during the first run.

```bash
python dsdb_cli.py update
```

### Filter PCDs.

Filters all PCDs. Here it is advised to use the Python-API for more space of expression.

```bash
python dsdb_cli.py filterpcds
```

### Sort PCDs.

Sorts all PCDs. Here it is advised to use the Python-API for more space of expression.

```bash
python dsdb_cli.py sortpcds
```

### Reject a QR-Code.

If you want to tag files and labelling them unfit for training, you can reject them via their QR-code.

```bash
python dsdb_cli.py rejectqrcode SAM-GOV-041
```

### Accept a QR-Code.

If you want to untag files via their QR-code, you can use this command.

```bash
python dsdb_cli.py acceptqrcode SAM-GOV-041
```

### List all rejected files.

It makes sense to be able to fetch all the rejected files. 

```bash
python dsdb_cli.py list rejected
```

### Create a filtered and preprocessed dataset.

In order to create a dataset from the database for training you have to do preprocessing. This filters the whole ETL-data and only uses valid data.

```bash
python dsdb_cli.py preprocess
```

## How to use Python-API

The Python-API is more powerful then the CLI. Especially when it comes to filtering and sorting. Note that there is also a Jupyter-notebook for demoing purposes called "python-api-demo.ipynb". Please have a look at it.

### Filter PCDs.

### Sort PCDs.