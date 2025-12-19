class Solution:
    def convert(self, s: str, numRows: int) -> str:
        zag_len = numRows - 2
        zigzag_idxs = [0]
        zig = []
        zag = []
        while True:
            zigzag_idxs.append(zigzag_idxs[-1] + numRows)
            zig.append(s[zigzag_idxs[-2] : zigzag_idxs[-1]])
            if zigzag_idxs[-1] > len(s):
                break
            zigzag_idxs.append(zigzag_idxs[-1] + zag_len)
            zag.append(s[zigzag_idxs[-2] : zigzag_idxs[-1]])
            if zigzag_idxs[-1] > len(s):
                break

        final = ""
        for i, chs in enumerate(zig):
            final += chs[0]
            zig[i] = zig[i][1:]

        for i in range(zag_len):
            for j in range(zag_len + 1):
                final += zig[i][j]
                final += zag[i][::-1][j]

        assert True


if __name__ == "__main__":

    sol = Solution()
    print(sol.convert("123456789", 3))
