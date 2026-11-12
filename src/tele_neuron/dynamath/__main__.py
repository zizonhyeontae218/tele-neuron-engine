"""DynaMath training, stateful character generation and live tick CLI."""
import argparse
import json

from .core import DynaMath
from .runtime import Session
from .training import benchmark, dump, experiment, setup, train_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("experiment", help="run bounded experiment 001")
    exp.add_argument("--root", default=".")
    train = sub.add_parser("train", help="character TBPTT; --steps is total, including resumed steps")
    train.add_argument("--train", default="data/dynamath/train.txt")
    train.add_argument("--validation", default="data/dynamath/validation.txt")
    train.add_argument("--output", required=True)
    train.add_argument("--resume")
    train.add_argument("--steps", type=int, default=800)
    for name in ("generate", "chat", "tick", "benchmark"):
        cli = sub.add_parser(name)
        cli.add_argument("--model", default="models/dynamath/characters.pt")
        cli.add_argument("--state-in")
        cli.add_argument("--state-out")
        if name in ("generate", "chat"):
            cli.add_argument("--max-chars", type=int, default=120)
            cli.add_argument("--max-ticks", type=int, default=2000)
        if name == "generate":
            cli.add_argument("--prompt", default="user: hi\nbot:")
        if name == "tick":
            cli.add_argument("--count", type=int, default=100)
            cli.add_argument("--seconds", type=float)
            cli.add_argument("--hz", type=float, default=100)
    args = parser.parse_args()
    setup(43)
    if args.command == "experiment":
        experiment(args.root)
        return
    if args.command == "train":
        _, result = train_text(args.train, args.validation, args.output,
                               steps=args.steps, resume=args.resume)
        dump(args.output + ".metrics.json", result)
        print(json.dumps(result, indent=2))
        return
    if args.state_in:
        session = Session.load(args.state_in)
    else:
        model, _ = DynaMath.load(args.model)
        session = Session(model)
    try:
        if args.command == "generate":
            print(session.generate(args.prompt, args.max_chars, args.max_ticks))
        elif args.command == "chat":
            print("DynaMath 1 character demo; /quit exits. Vocabulary is small.")
            while True:
                try:
                    message = input("you> ")
                except EOFError:
                    break
                if message == "/quit":
                    break
                prompt = "user: " + message.lower() + "\nbot:"
                if any(char not in session.ids for char in prompt):
                    print("unsupported character in input")
                    continue
                print("model> " + session.generate(prompt, args.max_chars, args.max_ticks))
        elif args.command == "tick":
            events = (list(session.realtime(args.seconds, args.hz)) if args.seconds is not None
                      else session.tick(args.count))
            print(json.dumps({"ticks": session.ticks, "phase": session.ticks % 5,
                              "events": "".join(events)}))
        else:
            print(json.dumps(benchmark(session.model), indent=2))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        if args.state_out:
            session.save(args.state_out)


if __name__ == "__main__":
    main()
