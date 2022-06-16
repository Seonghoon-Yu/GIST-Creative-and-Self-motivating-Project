import torch
from utils import get_recommends, post_process_recommends
from get_user_id import get_solved_problem, mapping_problems, sample_negative, get_user_id

device = 'cuda' if torch.cuda.is_available() else 'cpu'

user_name = input("유저 닉네임을 입력하세요: ")

print('Get user name: {}'.format(user_name))

### Get solved problems
print('Searching solved problems by user: {}'.format(user_name))
num_neg = 14000

solved_problems = get_solved_problem(user_name)
mapped_problems = mapping_problems(solved_problems)
problem_input = sample_negative(mapped_problems, num_neg)
user_id, user_id_aug = get_user_id(user_name, num_neg)

### Create model
print('Load pre-trained model')
n_model = 'NeuMF-end'
model_path = './checkpoint/'
NeuMF_model_path = model_path + 'NeuMF.pth'

GMF_model = None
MLP_model = None

model = torch.load(NeuMF_model_path, map_location=torch.device(device)).to(device)

model.to(device)
model.eval()

recommend_list = []

### Forward
print('Forward user data into AI model')
user = torch.tensor(user_id_aug).to(device)
item = torch.tensor(problem_input).to(device)

print('Get recommends')
recommends = get_recommends(user, item, model)
recommends = post_process_recommends(recommends, mapped_problems)

print('Recommend List')
for recommend in recommends:
    print('https://www.acmicpc.net/problem/{}'.format(recommend))














