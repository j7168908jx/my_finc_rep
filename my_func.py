from typing import List
import time


class QuickMysql:
    def __init__(self, connection=None, **kwargs):
        """
        setup a mysql connection and make easy query
        :param connection: a list/tuple including (host, user, password, database), database can be None.
        :param kwargs: may be dict that indicate each param: host, username, password, database
        """
        import pymysql
        if len(connection) == 0:
            try:
                connection = [kwargs.get('host'),
                              kwargs.get('username'),
                              kwargs.get('password'),
                              kwargs.get('database')
                              ]
            except KeyError:
                raise KeyError('Unable to get params!')

        pymysql.connections.DEFAULT_CHARSET = 'utf8'
        self.db = pymysql.connect(host=connection[0], user=connection[1], password=connection[2], database=connection[3],
                                  charset='utf8')

    def query_no_return(self, sql_: str):
        """

        :param sql_: sql script
        """
        cursor = self.db.cursor()
        cursor.execute(sql_)

    def query_with_return(self, sql_: str) -> List[tuple]:
        """

        :param sql_: sql script
        :return: result list
        """
        cursor = self.db.cursor()
        cursor.execute(sql_)
        fetch_ = cursor.fetchall()
        fetch_list_ = []
        for i_ in fetch_:
            fetch_list_.append(i_)
        return fetch_list_

    def query(self, sql_: str, have_return: bool):
        """

        :param sql_: sql script
        :param have_return: boolean value, true if the sql script has something to return.
        :return: result or simply true if nothing to return.
        """
        cursor = self.db.cursor()
        cursor.execute(sql_)
        if have_return:
            fetch_ = cursor.fetchall()
            fetch_list_ = []
            for i_ in fetch_:
                fetch_list_.append(i_)
            return fetch_list_
        else:
            return True


class Timer:

    def __init__(self, repeat_times=1):
        """

        :param repeat_times: the total loop times if this is used to time a loop procedure
        """
        self.total = repeat_times
        self.start()

    def start(self):
        """
        to start the timer
        """
        self._start = time.time()
        self.record = []
        self.last_avg_cost = -1
        self.count = 0

    def timeit(self) -> float:
        """

        :return: the time elapsed since the construction of this instance or the last time using 'this.start'
        """
        try:
            now = time.time()
            elapse = now - self._last
            self._last = now
        except AttributeError:
            self._last = time.time()
            elapse = self._last - self._start
        self.record.append(elapse)
        self.count += 1
        return round(elapse, 2)

    def predict_remaining(self, use_new=False) -> float:
        """
        to predict the remaining time of the loop, please set the total to the total loops
        :param use_new: re-calc the average loop time every time -- slower !!
        :return: the predicted remaining time if the total is bigger than count and the count is not 0
        """
        if self.count == 0 or self.count >= self.total:
            return -1
        # avg_cost = sum(self.record) / self.count
        if self.last_avg_cost == -1 or use_new:
            new_avg_cost = sum([pow(2, -i) * self.record[-i] for i in range(1, self.count+1)]) + pow(2, -self.count) * self.record[0]
            self.last_avg_cost = new_avg_cost
        else:
            new_avg_cost = 0.5 * (self.last_avg_cost + self.record[-1])
            self.last_avg_cost = new_avg_cost
        remaining = new_avg_cost * (self.total - self.count)
        time_unit = "s"
        if remaining > 600:
            remaining /= 60
            time_unit = "min"
            if remaining > 600:
                remaining /= 60
                time_unit = "h"
                
        used_time = time.time() - self._start
        time_unit2 = "s"
        if used_time > 600:
            used_time /= 60
            time_unit2 = "min"
            if used_time > 600:
                used_time /= 60
                time_unit2 = "h"

        result = "Now {}/{}, estimated remaining: {:.3f}{}, loop cost: {:.4f}s, totally used time: {:.3f}{}".format(
            self.count-1, self.total, remaining, time_unit, new_avg_cost, used_time, time_unit2)
        print(result)
        return remaining

def select_useful_data(list_: List[List], useful_index_list_: List[int]) -> List[List]:
    """This func is used to make slices of list and return the selected index's data of the whole list.

    :param list_: data list, shape like[m, n], e.g. [ [1, 2, 3], [4, 5, 6] ]
    :param useful_index_list_: index list, shape like[k], k<=n, e.g. [1, -1]
    :return: useful_list, for the example above, shape like [m, k], e.g. [ [2, 3], [5, 6] ]
    """
    output_list_ = []
    for i_ in list_:
        useful_data_list_ = []
        for j_ in useful_index_list_:
            useful_data_list_.append(i_[j_])
        output_list_.append(useful_data_list_)
    return output_list_


def import_data_to_list(file_path_: str, have_description_=True, encoding='ANSI', delimiter=',', strip=False) -> List[List]:
    """This func is used to import data from a csv or tsv file
    If data has title, then delete it
    :param file_path_: a csv/tsv file path, str type
    :param have_description_: boolean, if the data has title then we will delete it
    :param encoding: encoding of the file
    :param delimiter: the default delimiter
    :param strip: whether needed to use strip()
    :return: data list in the file, with shape[m, n]
    """
    file_type_ = file_path_[-3:]

    if file_type_ == 'csv':
        spliter_ = ','
    elif file_type_ == 'tsv':
        spliter_ = '\t'
    else:
        spliter_ = delimiter

    try:
        file_ = open(file_path_, 'r+', encoding=encoding)
        data_list_ = file_.readlines()
    except UnicodeDecodeError:
        file_.close()
        file_ = open(file_path_, 'r+', encoding='utf-8')
        data_list_ = file_.readlines()
    if have_description_:
        del data_list_[0]
    data_output_ = []
    for i_ in data_list_:
        i_ = i_.split(spliter_)
        for j_ in range(len(i_)):
            try:
                if i_[j_][0] == '"':
                    i_[j_] = i_[j_][1:-1]
            except IndexError:
                pass


        if i_[-1][-2:] == '\r\n':
            i_[-1] = i_[-1][:-2]
        elif i_[-1][-1:] == '\n':
            i_[-1] = i_[-1][:-1]
        else:
            pass

        if strip:
            for j_ in range(len(i_)):
                i_[j_] = i_[j_].strip()
        data_output_.append(i_)
        file_.close()
    for i_ in range(len(data_output_)):
        for j_ in range(len(data_output_[i_])):
            try:
                data_output_[i_][j_] = float(data_output_[i_][j_])
                # data_output_[i_][j_] = eval(data_output_[i_][j_])
            except ValueError:
                pass
    return data_output_


def export_data_to_new_csv(file_path_: str, data_list_: List[List]):
    """

    :param file_path_: the file path of your new file
    :param data_list_: the data list
    :return:
    """
    file_ = open(file_path_, 'w+', encoding='utf-8')
    data_write_ = []
    for i_ in data_list_:
        for j_ in range(len(i_)):
            if not isinstance(i_[j_], str):
                i_[j_] = str(i_[j_])

        data_write_.append(','.join(i_) + '\n')

    file_.writelines(data_write_)
    file_.close()

