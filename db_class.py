import pymysql

def text_prepare(value):
    """
    Prepare text field in such a vay that it will not cause problems whe inserted into SQL quiery
    """

    if type(value) == str:
        return value.replace('"',"'").strip()
    else:
        return value

class StorageDatabase:

    def __init__(self, db_host, db_port, db_name, db_user, sql_password):
        sql_host = db_host
        sql_user = db_user
        self.__connection__ = pymysql.connect(host=sql_host,
                                              user=sql_user,
                                              password=sql_password,
                                              cursorclass=pymysql.cursors.DictCursor)
        sql = f'USE {db_name}'
        self.sql_exec(sql)

        print("Database opened sucsessfully")

    def sql_exec(self,sql,show = 'n', use = 'u', *kwargs):
        """
        execute SQL quierty. if show == 'y' print the result to the standard output
        """

        with self.__connection__.cursor() as cursor:

            try:
                cursor.execute(sql)
            except pymysql.err.ProgrammingError as er:
                print(f"\n\n SQL failed on \n {sql}")
                raise pymysql.err.ProgrammingError(er)
            except pymysql.err.DataError as er:
                print(f"\n\n SQL failed on \n {sql}")
                if er.find("1406") >= 0:
                    print(f"Text too long, field not saved")
                else:
                    raise pymysql.err.DataError(er)
            except pymysql.err.OperationalError as er:
                print(f"\n\n SQL failed on \n {sql}")
                raise pymysql.err.OperationalError(er)
            result = cursor.fetchall()
            res = []
            for item in result:
                res.append(item)
                if show == 's':
                    print(item)
            return res


    def table_add_row(self, table, data_items):
        """
        Add row into the table
        """

        names = []
        values = []
        for key, value in data_items.items():
            names.append(key)
            values.append(value)
        col_names = "".join([f'{name}, ' for name in names]).strip(', ')
        col_values = "".join([f'"{text_prepare(value)}", ' for value in values]).strip(', ')
        sql = f"INSERT INTO {table} ({col_names}) VALUES ({col_values})"

        self.sql_exec(sql)

    #
    # def table_update_row_return_id(self, table, column, value, data_items):
    #     """
    #     If the row with given value in given column does not exist,
    #     adds the row to the table with the given set of values and returns its ID.
    #     Otherwise returns the ID of the existing row.
    #     Does NOT update existing rows
    #
    #     """
    #     row = self.__table_find_row__(table, column, value)
    #     if len(row) > 0:
    #         return row[0]['ID']
    #     else:
    #         self.table_add_row(table, data_items)
    #         sql = f"SELECT MAX(ID) AS ID FROM {table}"
    #         return self.sql_exec(sql)[0]['ID']
    #
    #
    # def __table_find_row__(self, table, column, value):
    #     """
    #     Finds row in the table with given value in the given row, returns ID
    #     """
    #     sql = f'SELECT ID FROM {table}  WHERE {column} = "{text_prepare(value)}"'
    #     res = self.sql_exec(sql)
    #     return res
    #
    def table_get_value_with_ID(self, table, ID, columns):
        """
        Returns value in the row by ID

        """
        fields = ",".join(columns)
        sql = f'SELECT {columns} FROM {table}  WHERE ID = {ID}'
        res = self.sql_exec(sql)
        if len(res) > 0:
            return res[0]
        else:
            return []
    #
    # def table_find_values(self, table, column, value, limit = 1):
    #     """
    #     Returns value in the row by ID
    #
    #     """
    #     sql = f'SELECT * FROM {table}  WHERE {column} = "{value}" LIMIT {limit}'
    #     res = self.sql_exec(sql)
    #     if len(res) > 0:
    #         return res
    #     else:
    #         return []
    #
    # def current_no_of_records(self, table = 'job_card'):
    #     """
    #     Returns the number of the job announcements recorded in the database at the moment
    #     """
    #     sql = f"SELECT MAX(ID) AS ID FROM {table}"
    #     return self.sql_exec(sql)[0]['ID']
    #
    # def table_update_row(self, table, ID, column, value):
    #     """
    #     Updates the row with the given value by ID
    #
    #     """
    #     sql = f'UPDATE {table} SET {column} = "{text_prepare(value)}" WHERE ID = {ID}'
    #     res = self.sql_exec(sql)
    #     if len(res) > 0:
    #         return res[0]
    #     else:
    #         return []
    #
    def db_commit(self):
        self.__connection__.commit()


if __name__ == "__main__":

    db = StorageDatabase("sql11.freemysqlhosting.net","3306","sql11593194","sql11593194","2CPjwjQHDQ")