from agent import Agent
from dataloader import iVQADataloader, IVQAConfig


def run_user(agent: Agent):
    agent.run()


def evaluate_ivqa(agent: Agent, sample_num=100):
    config = IVQAConfig()
    dataloader_wrapper = iVQADataloader(config=config, batch_size=1, shuffle=False)
    loader = dataloader_wrapper.get_loader()

    qa_num = 0
    correct_num = 0
    reasonable_num = 0
    for batch in loader:
        if qa_num >= sample_num:
            break
        qa_num += 1

        answers = batch["answers"]
        video_path = batch["video_path"][0]
        question = batch["question"][0] + f"video_path: {video_path}"

        agent.api_executor.init_memory()
        answer, _ = agent.run_single_step(question)
        print(f"Question: {question}")
        print(f"Answer: {answer}, gt: {answers}")
        hit = [1 if ans[0].lower() in answer.lower() else 0 for ans in answers]
        if sum(hit) >= 3:
            correct_num += 1
        if sum(hit) > 0:
            reasonable_num += 1

        print(
            f"Accuracy: {correct_num / qa_num:.2f}, Reasonable: {reasonable_num / qa_num:.2f}"
        )
        print(
            f"total qa num: {qa_num}, correct num: {correct_num}, reasonable num: {reasonable_num}"
        )
        print("--------------------------------------------------")


if __name__ == "__main__":

    agent = Agent(
        sample_prompt="what does the person do after scooping the food near the beginning of the video 'videos/ivqa_example2.webm'?"
    )
    # run_user(agent)
    evaluate_ivqa(agent, sample_num=100)
    # 예시 입력
