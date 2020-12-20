import torch

hiddens = torch.randn(3, 4, 5)
print("hiddens", hiddens)
# print(hiddens)
# cur_hiddens = hiddens[0].data
# print("cur_hiddens", cur_hiddens)
#
# pos = torch.tensor(range(1, 4))
# cur_hiddens = cur_hiddens[pos, :]
# print(cur_hiddens.size())
# average_hiddens = torch.mean(cur_hiddens, dim=0)
# print(average_hiddens.size())

# batch = []
# batch.append(torch.randn(3))
# batch.append(torch.randn(3))
# print(batch)
# batch_hiddens = torch.ones(2, 3)
# i = 0
# for i in range(2):
#     batch_hiddens.data[i, :] = batch[i].data
# print(batch_hiddens)
# print(batch_hiddens.size())
extract_hiddens = torch.randn(3, 5)
print("extract_hiddens", extract_hiddens)
extract_hiddens = extract_hiddens.unsqueeze(1).expand(3, 4, 5)
print("after extract_hiddens", extract_hiddens.size())
hiddens = torch.cat([hiddens, extract_hiddens], dim=-1)
print("cat hiddens", hiddens.size())
