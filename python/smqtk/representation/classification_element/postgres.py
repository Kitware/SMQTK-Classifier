from six.moves import cPickle

from smqtk.exceptions import NoClassificationError
from smqtk.representation import ClassificationElement


# Try to import required modules
try:
    import psycopg2
except ImportError:
    psycopg2 = None


__author__ = "paul.tunison@kitware.com"


class PostgresClassificationElement (ClassificationElement):
    """
    PostgreSQL database backed classification element.

    Requires a table of at least 3 fields (column names configurable):

        - type-name :: text
        - uuid :: text
        - classification-binary :: bytea

    """

    UPSERT_TABLE_TMPL = ' '.join("""
        CREATE TABLE IF NOT EXISTS {table_name:s} (
          {type_col:s} TEXT NOT NULL
          {uuid_col:s} TEXT NOT NULL,
          {classification_col:s} BYTEA NOT NULL,
          PRIMARY KEY ({type_col:s}, {uuid_col:s})
        );
    """.split())

    # Known psql version compatibility: 9.4
    SELECT_TMPL = ' '.join("""
        SELECT {classification_col:s}
          FROM {table_name:s}
          WHERE {type_col:s} = %(type_val)s
            AND {uuid_col:s} = %(uuid_val)s
        ;
    """.split())

    # Known psql version compatibility: 9.4
    UPSERT_TMPL = ' '.join("""
        WITH upsert AS (
          UPDATE {table_name:s}
            SET {classification_col:s} = %(classification_val)s
            WHERE {type_col:s} = %(type_val)s
              AND {uuid_col:s} = %(uuid_val)s
            RETURNING *
          )
        INSERT INTO {table_name:s}
          ({type_col:s}, {uuid_col:s}, {classification_col:s})
          SELECT %(type_val)s, %(uuid_val)s, %(classification_val)s
            WHERE NOT EXISTS (SELECT * FROM upsert);
    """.split())

    @classmethod
    def is_usable(cls):
        if psycopg2 is None:
            cls.get_logger().warning("Not usable. Requires psycopg2 module")
            return False
        return True

    def __init__(self, type_name, uuid,
                 table_name='classifications',
                 type_col='type_name', uuid_col='uid',
                 classification_col='classification',
                 db_name='postgres', db_host=None, db_port=None, db_user=None,
                 db_pass=None, pickle_protocol=-1, create_table=True):
        """
        Initialize new PostgresClassificationElement attached to some database
        credentials.

        We require that storage tables treat uuid AND type string columns as
        primary keys. The type and uuid columns should be of the ``text`` type.
        The binary column should be of the ``bytea`` type.

        Default argument values assume a local PostgreSQL database with a table
        created via the
        ``etc/smqtk/postgres/classification_element/example_table_init.sql``
        file (relative to the SMQTK source tree or install root).

        NOTES:
            - Not all uuid types used here are necessarily of the ``uuid.UUID``
              type, thus the recommendation to use a ``text`` type for the
              column. For certain specific use cases they may be proper
              ``uuid.UUID`` instances or strings, but this cannot be generally
              assumed.

        :param type_name: Name of the type of classifier this classification was
            generated by.
        :type type_name: str

        :param uuid: Unique ID reference of the classification
        :type uuid: collections.Hashable

        :param table_name: String label of the database table to use.
        :type table_name: str

        :param uuid_col: The column label for classification UUID storage
        :type uuid_col: str

        :param type_col: The column label for classification type name storage.
        :type type_col: str

        :param classification_col: The column label for classification binary
            storage.
        :type classification_col: str

        :param db_host: Host address of the Postgres server. If None, we
            assume the server is on the local machine and use the UNIX socket.
            This might be a required field on Windows machines (not tested yet).
        :type db_host: str | None

        :param db_port: Port the Postgres server is exposed on. If None, we
            assume the default port (5423).
        :type db_port: int | None

        :param db_name: The name of the database to connect to.
        :type db_name: str

        :param db_user: Postgres user to connect as. If None, postgres
            defaults to using the current accessing user account name on the
            operating system.
        :type db_user: str | None

        :param db_pass: Password for the user we're connecting as. This may be
            None if no password is to be used.
        :type db_pass: str | None

        :param pickle_protocol: Pickling protocol to use. We will use -1 by
            default (latest version, probably binary).
        :type pickle_protocol: int

        :param create_table: If this instance should try to create the storing
            table before actions are performed against it. If the configured
            user does not have sufficient permissions to create the table and it
            does not currently exist, an exception will be raised.
        :type create_table: bool

        """
        super(PostgresClassificationElement, self).__init__(type_name, uuid)

        self.table_name = table_name
        self.type_col = type_col
        self.uuid_col = uuid_col
        self.classification_col = classification_col

        self.db_name = db_name
        self.db_host = db_host
        self.db_port = db_port
        self.db_user = db_user
        self.db_pass = db_pass

        self.pickle_protocol = pickle_protocol
        self.create_table = create_table

    def get_config(self):
        return {
            "table_name": self.table_name,
            "type_col": self.type_col,
            "uuid_col": self.uuid_col,
            "classification_col": self.classification_col,

            "db_name": self.db_name,
            "db_host": self.db_host,
            "db_port": self.db_port,
            "db_user": self.db_user,
            "db_pass": self.db_pass,

            "pickle_protocol": self.pickle_protocol,
            "create_table": self.create_table,
        }

    def _get_psql_connection(self):
        """
        :return: A new connection to the configured database
        :rtype: psycopg2._psycopg.connection
        """
        return psycopg2.connect(
            database=self.db_name,
            user=self.db_user,
            password=self.db_pass,
            host=self.db_host,
            port=self.db_port,
        )

    def _ensure_table(self, cursor):
        """
        Execute on psql connector cursor the table create-of-not-exists query.

        :param cursor: Connection active cursor.

        """
        if self.create_table:
            q_table_upsert = self.UPSERT_TABLE_TMPL.format(**dict(
                table_name=self.table_name,
                type_col=self.type_col,
                uuid_col=self.uuid_col,
                classification_col=self.classification_col,
            ))
            cursor.execute(q_table_upsert)

    def has_classifications(self):
        """
        :return: If this element has classification information set.
        :rtype: bool
        """
        try:
            return bool(self.get_classification())
        except NoClassificationError:
            return False

    def get_classification(self):
        """
        Get classification result map, returning a label-to-confidence dict.

        We do no place any guarantees on label value types as they may be
        represented in various forms (integers, strings, etc.).

        Confidence values are in the [0,1] range.

        :raises NoClassificationError: No classification labels/confidences yet
            set.

        :return: Label-to-confidence dictionary.
        :rtype: dict[collections.Hashable, float]

        """
        q_select = self.SELECT_TMPL.format(**dict(
            table_name=self.table_name,
            type_col=self.type_col,
            uuid_col=self.uuid_col,
            classification_col=self.classification_col,
        ))
        q_select_values = {
            "type_val": self.type_name,
            "uuid_val": str(self.uuid)
        }

        conn = self._get_psql_connection()
        cur = conn.cursor()
        try:
            self._ensure_table(cur)
            cur.execute(q_select, q_select_values)
            r = cur.fetchone()
            # For server cleaning (e.g. pgbouncer)
            conn.commit()

            if not r:
                raise NoClassificationError("No PSQL backed classification for "
                                            "label='%s' uuid='%s'"
                                            % (self.type_name, str(self.uuid)))
            else:
                b = r[0]
                c = cPickle.loads(str(b))
                return c
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

    def set_classification(self, m=None, **kwds):
        """
        Set the whole classification map for this element. This will strictly
        overwrite the entire label-confidence mapping (vs. updating it)

        Label/confidence values may either be provided via keyword arguments or
        by providing a dictionary mapping labels to confidence values.

        :param m: New labels-to-confidence mapping to set.
        :type m: dict[collections.Hashable, float]

        :raises ValueError: The given label-confidence map was empty.

        """
        m = super(PostgresClassificationElement, self)\
            .set_classification(m, **kwds)

        q_upsert = self.UPSERT_TMPL.strip().format(**{
            "table_name": self.table_name,
            "classification_col": self.classification_col,
            "type_col": self.type_col,
            "uuid_col": self.uuid_col,
        })
        q_upsert_values = {
            "classification_val":
                psycopg2.Binary(cPickle.dumps(m, self.pickle_protocol)),
            "type_val": self.type_name,
            "uuid_val": str(self.uuid),
        }

        conn = self._get_psql_connection()
        cur = conn.cursor()
        try:
            self._ensure_table(cur)
            cur.execute(q_upsert, q_upsert_values)
            cur.close()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()
