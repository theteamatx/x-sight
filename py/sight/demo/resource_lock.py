from readerwriterlock import rwlock

class RWLockDictWrapper:

    # Shared with every instance
    __shared_state = {}

    def __init__(self):
        self.__dict__ = self.__shared_state
        self.lock = rwlock.RWLockWrite()
        self.resource = {}

    def get(self):
      with self.lock.gen_rlock():
        return self.resource

    def get_for_key(self, key):
      with self.lock.gen_rlock():
        return self.resource.get(key,None)

    def set_for_key(self, key, value):
      with self.lock.gen_wlock():
        self.resource[key] = value

    def update(self, mapping):
      with self.lock.gen_wlock():
        self.resource.update(mapping)

  