import { useEffect, useRef, useState } from "react";
import Layout from "~/components/layout";
import Header from "~/components/ui/header";

interface ConversationProps {
	type: "single" | "compare";
	question?: string;
	reply?: string;
	replyCompare?: string[];
}

export default function Chat() {
	const [input, setInput] = useState<string>("");
	// const [chats, setChats] = useState<string[]>([]);
	const [typeOfReply, setTypeOfReply] =
		useState<ConversationProps["type"]>("single");
	const [chats, setChats] = useState<ConversationProps[]>([]);

	const chatEndsRef = useRef<HTMLDivElement>(null);

	const replies: string[] = [
		"Ea adipisicing deserunt in proident ea cupidatat ullamco commodo et laborum.",
		"Reprehenderit ullamco culpa ea cillum deserunt. Deserunt in nulla dolor nulla Lorem esse est nulla anim est. Consequat eu sunt aliquip proident. Cillum ea nulla aliquip velit commodo cupidatat elit deserunt sunt amet in. Incididunt nisi et ex sint do nostrud magna consequat exercitation nisi exercitation laboris reprehenderit. Laboris ut consequat dolor occaecat deserunt commodo exercitation laborum aute. Occaecat veniam nostrud dolor nulla occaecat elit duis qui reprehenderit pariatur.",
		"Dolor fugiat excepteur eiusmod et ea fugiat reprehenderit est ut nisi.",
		"Et tempor enim amet veniam elit.",
		"Laborum laboris consectetur aliqua sit reprehenderit laboris labore et dolor ad fugiat et commodo. Cillum adipisicing voluptate aute veniam nulla commodo nisi reprehenderit eu. Exercitation consectetur pariatur irure officia ea exercitation cillum sit duis esse amet incididunt sit magna. Aliqua commodo ex ipsum sunt mollit pariatur duis cillum do laboris tempor nulla velit incididunt. Tempor nulla anim nulla fugiat labore est duis officia eiusmod irure esse dolor.",
	];

	const repliesCompare: string[] = [
		"Lorem labore ut sint fugiat fugiat nulla laborum labore sit laboris deserunt veniam nostrud laborum. Ad deserunt duis sit culpa dolore irure labore esse laborum ad. Aliquip est deserunt nostrud commodo irure aliquip magna aliqua veniam duis. Elit irure anim consequat pariatur laborum nulla excepteur proident mollit aliqua id ipsum. Nulla ut et ipsum do minim est ut consequat nostrud sunt labore excepteur. Irure reprehenderit aute veniam sunt aute sint non aliqua. Nostrud quis do tempor aute do. Tempor velit laborum aliquip voluptate eu excepteur. Anim consectetur enim nulla nisi nisi laborum exercitation tempor veniam dolore nisi quis. Dolor sunt deserunt fugiat elit ut cillum ipsum anim enim id. Deserunt ullamco velit amet mollit eu non minim ex occaecat ullamco aute veniam.",
		"Duis pariatur deserunt aute aute veniam ut. Proident veniam reprehenderit do culpa ad labore ea ad culpa mollit quis laborum. Eiusmod cillum dolor fugiat ex tempor est anim sint anim aliquip mollit commodo. Qui quis ut veniam consectetur. Mollit veniam mollit commodo in deserunt cillum qui laboris officia non ex. Consequat non ad nostrud nulla ad consectetur commodo sint reprehenderit. Eu veniam id anim nulla dolore quis pariatur. Fugiat aute nisi sint eiusmod eu eiusmod ullamco exercitation cupidatat fugiat. Officia ex enim officia pariatur.",
	];

	const submitHandler = (e: React.FormEvent<HTMLFormElement>) => {
		e.preventDefault();
		const trimmed = input.trim();
		if (trimmed === "") return;

		// Fixed: Ensure 'type' matches the interface
		const newChat: ConversationProps = trimmed.includes("compare")
			? {
					type: "compare",
					question: trimmed,
					replyCompare: repliesCompare,
				}
			: {
					type: "single",
					question: trimmed,
					reply: replies[Math.floor(Math.random() * replies.length)], // Makes it pick a random reply
				};

		// Fixed: Append new chat to the bottom of the array
		setChats((prev) => [...prev, newChat]);
		setInput("");
		console.log("New chat added:", newChat); // Debugging line
	};

	const placeholder = [
		"Hi, how can I help you?",
		"What office-related tasks can I assist with?",
		"Need help with scheduling a meeting?",
		"How can I help with your office work?",
		"Do you need assistance with office administration?",
		"What office matter can I help you with today?",
	];

	// Efek untuk menggulir ke bawah setiap kali array 'chats' berubah
	useEffect(() => {
		chatEndsRef.current?.scrollIntoView({ behavior: "smooth" });
	}, [chats]);

	return (
		<Layout>
			{/* <div className="w-full h-full flex flex-col overflow-y-auto bg-sidebar rounded-lg"> */}
			<Header title="Chat" />
			<div className="pl-4 pr-4 pb-4">
				<div className="flex flex-col items-center justify-center">
					{/* Chat view */}
					<div className="flex-1 max-h-[83vh] min-h-[83vh] h-full overflow-y-auto scrollbar-none pt-5">
						{chats.map((chat, index) => (
							<div>
								<div
									key={index}
									className="flex flex-col gap-2 mb-4">
									<div className="flex w-full max-w-4xl min-w-4xl">
										<div className="self-end flex justify-end items-end w-full">
											<p className="bg-slate-200 text-black px-4 py-2 rounded-2xl max-w-[70%]">
												{chat.question}
											</p>
										</div>
									</div>

									{chat.type === "single" ? (
										<div className="flex w-full max-w-4xl min-w-4xl">
											<div className="self-start px-4 py-2 rounded-2xl flex w-full">
												<p>{chat.reply}</p>
											</div>
										</div>
									) : (
										<div className="flex w-full max-w-4xl mx-auto">
											<div className="self-start px-4 py-2 rounded-2xl flex w-full">
												<div className="grid grid-cols-2 gap-4">
													{" "}
													{/* Added grid columns for comparison */}
													{chat.replyCompare?.map(
														(reply, idx) => (
															<div
																key={idx}
																className="flex flex-col gap-2">
																<div className="bg-blue-50 p-4 rounded-xl border border-blue-100">
																	<p>
																		{reply}
																	</p>
																</div>
																<div className="flex gap-2 p-2 items-center justify-end">
																	<p className="text-sm text-gray-500">
																		Is the
																		answer
																		helpful?
																	</p>
																	<div className="flex items-center gap-2">
																		{/* Option Yes */}
																		<div className="flex items-center">
																			<input
																				type="radio"
																				id={`helpful-${index}-${idx}-yes`}
																				name={`helpful-${index}-${idx}`}
																				value="yes"
																				className="peer hidden"
																			/>
																			<label
																				htmlFor={`helpful-${index}-${idx}-yes`}
																				className="cursor-pointer select-none rounded-full border border-blue-100 bg-blue-50 px-3 py-1 text-sm transition hover:border-blue-200 hover:bg-blue-100 peer-checked:border-blue-300 peer-checked:bg-blue-200 peer-checked:font-semibold">
																				Yes
																			</label>
																		</div>

																		{/* Option No */}
																		<div className="flex items-center">
																			<input
																				type="radio"
																				id={`helpful-${index}-${idx}-no`}
																				name={`helpful-${index}-${idx}`}
																				value="no"
																				className="peer hidden"
																			/>
																			<label
																				htmlFor={`helpful-${index}-${idx}-no`}
																				className="cursor-pointer select-none rounded-full border border-blue-100 bg-blue-50 px-3 py-1 text-sm transition hover:border-blue-200 hover:bg-blue-100 peer-checked:border-blue-300 peer-checked:bg-blue-200 peer-checked:font-semibold">
																				No
																			</label>
																		</div>
																	</div>
																</div>
															</div>
														),
													)}
												</div>
											</div>
										</div>
									)}
								</div>
							</div>
						))}
						<div ref={chatEndsRef} />
					</div>

					{/* input */}
					<div className="w-full max-w-4xl px-4">
						<div className="w-full">
							<form
								onSubmit={submitHandler}
								className="bg-white ring-1 ring-slate-200 rounded-full flex items-center gap-2 px-3 py-2">
								<input
									type="text"
									placeholder={`${placeholder[Math.floor(Math.random() * placeholder.length)]}`}
									className="w-full active:outline-none focus:outline-none"
									value={input}
									onChange={(e) => setInput(e.target.value)}
								/>
								<button type="submit">Send</button>
							</form>
						</div>
					</div>
				</div>
			</div>
			{/* </div> */}
		</Layout>
	);
}
