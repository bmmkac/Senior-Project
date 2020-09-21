import pymysql
import json

def getConfig(config_file='/home/mysql.config'):
    with open(config_file) as handle:
        config = json.loads(handle.read())
        return config


class DigiPathDbWrapper(object):
    def __enter__(self):
        class DigiPathDb:
            def __init__(self):
                db_config = getConfig()

                try:
                    print('connecting')
                    self.conn = pymysql.connect(db_config['host'], db_config['user'], db_config['pwd'], db_config['db'])
                    self.cursor = self.conn.cursor()
                    self.cursor.execute('SELECT VERSION()')
                    db_version = self.cursor.fetchone()
            
                except Exception as error:
                    print('Error: connection not established {}'.format(error))
                    self.conn = None
                    self.cursor = None

                else:
                    print('connection established\n{}'.format(db_version[0]))

            def close(self):
                self.conn.close()
                self.cursor.close()

            def query(self, query, args=None):
                try:
                    result = self.cursor.execute(query, args)
                except Exception as error:
                    print('error executing query "{}", error: {}'.format(query, error))
                    return None
                else:
                    return result, self.cursor.fetchall()

            def new_user(self, email, classification):
                cmd = "INSERT INTO `users` (`email`, `class`) VALUES (%s, %s)"
                res = self.query(cmd, (email, classification))
                if res:
                    self.conn.commit()

            def new_slide(self, slide_name, slide_path, classification='unknown'):
                cmd = "INSERT INTO `slides` (`name`, `path`, `class`) VALUES (%s, %s, %s)"
                res = self.query(cmd, (slide_name, slide_path, classification))
                if res:
                    self.conn.commit()

            def new_cell(self, cell_path, slide_name, tile_path, row, col):
                cmd = "INSERT INTO `cells` (`path`, `slide_name`, `tile_path`, `pixel_row`, `pixel_col`) VALUES (%s, %s, %s, %s, %s)"
                res = self.query(cmd, (cell_path, slide_name, tile_path, row, col))
                if res:
                    self.conn.commit()

            def new_cell_classification(self, cell_id,  user_id, classification):
                cmd = "INSERT INTO `cell_classes` (`cell_id`, `user_id`, `class`) VALUES (%s, %s, %s)"
                res = self.query(cmd, (cell_id, user_id, classification))
                if res:
                    self.conn.commit()

            def get_unclassified_cells(self):
                cmd = "SELECT cell_id, path FROM cells WHERE cell_id NOT IN (SELECT cell_id FROM cell_classes)"
                res = self.query(cmd)
                if res:
                    return res[1]

            def get_cell_classifications(self, cell_id):
                cmd = "SELECT class FROM cell_classes WHERE cell_id = %s"
                res = self.query(cmd, (cell_id))
                if res:
                    return res[1]
            
            def get_next_cell_ID(self):
                cmd = "SELECT MAX(cell_id) FROM cells"
                res = self.query(cmd)
                if res:
                    val = res[1]
                    id = val[0][0]
                    if id is None:
                        return 1
                    else:
                        return id + 1

        self.db_obj = DigiPathDb()
        return self.db_obj

    def __exit__(self, exc_type, exc_value, traceback):
        self.db_obj.close()
    