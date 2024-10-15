from typing import List


class PrefixTreeNode:
    def __init__(self):
        # словарь с буквами, которые могут идти после данной вершины
        self.children: dict[str, PrefixTreeNode] = {}
        self.is_end_of_word = False


class PrefixTree:
    def __init__(self, vocabulary: List[str]):
        """
        vocabulary: список всех уникальных токенов в корпусе
        """
        self.root = PrefixTreeNode()

        for word in vocabulary:
            self.insert(word)

    def insert(self, word):
        node = self.root
        for letter in word:
            if letter not in node.children:
                node.children[letter] = PrefixTreeNode()
            node = node.children[letter]

        node.is_end_of_word = True

    def search_prefix(self, prefix) -> List[str]:
        """
        Возвращает все слова, начинающиеся на prefix
        prefix: str – префикс слова (чзх че за символы сюда вставили неридабельные?))
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        return self.find_words(node, prefix)

    def find_words(self, node: PrefixTreeNode, prefix: str) -> List[str]:
        words = []
        if node.is_end_of_word:
            words.append(prefix)

        for char, child_node in node.children.items():
            words.extend(self.find_words(child_node, prefix + char))

        return words
