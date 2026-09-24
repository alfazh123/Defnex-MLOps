import { useState } from "react";

import ModalRunSFT from "~/components/modal/run-sft";
import { columnsTrainRun } from "~/components/table/columns/train-column";
import { DataTable } from "~/components/table/data-table";
import Header from "~/components/ui/header";

import { trainingRunsData } from "~/utils";

import {
	CardsThreeIcon,
	FireIcon,
	HourglassMediumIcon,
	SubtractIcon,
} from "@phosphor-icons/react/dist/ssr";
import clsx from "clsx";

export default function TrainRun() {
	const [modalSft, setModalSft] = useState(false);
	const [selectedRunId, setSelectedRunId] = useState<string>(
		trainingRunsData[0]?.runId || "",
	);
	const [selecttedStatus, setSelectedStatus] = useState<string>("ALL");
	const [filteredData, setFilteredData] = useState(trainingRunsData);

	const toggleModalSft = () => {
		setModalSft((prevState) => !prevState);
	};

	const completedTrain = trainingRunsData.filter(
		(run) => run.status === "COMPLETED",
	);
	const failedTrain = trainingRunsData.filter(
		(run) => run.status === "FAILED",
	);
	const runningTrain = trainingRunsData.filter(
		(run) => run.status === "RUNNING",
	);
	const pendingTrain = trainingRunsData.filter(
		(run) => run.status === "PENDING",
	);

	const handleStatusChange = (status: string) => {
		if (status === selecttedStatus) {
			setSelectedStatus("ALL");
			setFilteredData(trainingRunsData);
			return;
		}
		setSelectedStatus(status);
		setFilteredData(
			status === "ALL"
				? trainingRunsData
				: status === "PENDING" || status === "RUNNING"
					? trainingRunsData.filter(
							(run) =>
								run.status === "PENDING" ||
								run.status === "RUNNING",
						)
					: trainingRunsData.filter((run) => run.status === status),
		);
	};

	const cardTrainStatus = [
		{
			label: "All run",
			id: "ALL",
			value: trainingRunsData.length,
			className:
				"border border-sky-100 text-sky-900 rounded-3xl bg-sky-50/70 shadow-sm",
			idleClassName:
				"border border-sky-100/50 text-sky-900/50 rounded-3xl bg-sky-50/90",
			icon: (
				<CardsThreeIcon
					className="w-7 h-7 text-sky-500"
					weight="fill"
				/>
			),
			activeClass: "bg-sky-400",
			inActiveClass: "border border-3 border-sky-400/50",
		},
		{
			label: "Completed",
			id: "COMPLETED",
			value: completedTrain.length,
			className:
				"border border-emerald-100 text-emerald-900 rounded-3xl bg-emerald-50/70 shadow-sm",
			idleClassName:
				"border border-emerald-100/50 text-emerald-900/50 rounded-3xl bg-emerald-50/90",
			icon: (
				<SubtractIcon
					className="w-7 h-7 text-emerald-600"
					weight="fill"
				/>
			),
			activeClass: "bg-emerald-400",
			inActiveClass: "border border-3 border-emerald-400/50",
		},
		{
			label: "Failed Train",
			id: "FAILED",
			value: failedTrain.length,
			className:
				"border border-rose-100 text-rose-900 rounded-3xl bg-rose-50/70 shadow-sm",
			idleClassName:
				"border border-rose-100/50 text-rose-900/50 rounded-3xl bg-rose-50/90",
			icon: (
				<FireIcon
					className="w-7 h-7 text-rose-500"
					weight="fill"
				/>
			),
			activeClass: "bg-rose-400",
			inActiveClass: "border border-3 border-rose-400/50",
		},
		{
			label: "Pending",
			id: "PENDING",
			value: pendingTrain.length + runningTrain.length,
			className:
				"border border-amber-100 text-amber-900 rounded-3xl bg-amber-50/70 shadow-sm",
			idleClassName:
				"border border-amber-100/50 text-amber-900/50 rounded-3xl bg-amber-50/90",
			icon: (
				<HourglassMediumIcon
					className="w-7 h-7 text-amber-500"
					weight="fill"
				/>
			),
			activeClass: "bg-amber-400",
			inActiveClass: "border border-3 border-amber-400/50",
		},
	];

	return (
		<div className="flex flex-col min-h-screen bg-[#f3f4f6]/60 p-4 lg:p-6 text-slate-800 font-sans">
			<Header
				title="SFT Execution & Training Runs"
				description="Unsloth-backed Supervised Fine-Tuning jobs and mock worker tracking"
				sft
				modalSft={toggleModalSft}
			/>
			<div className="px-2 pb-10 max-w-7xl w-full mx-auto">
				<div className="grid md:grid-cols-4 grid-cols-2 my-6 gap-4">
					{cardTrainStatus.map((status) => (
						<div
							key={status.label}
							className={`${selecttedStatus === status.id ? status.className : status.idleClassName} px-5 py-4 flex flex-col justify-between transition-all hover:scale-[1.01] h-36`}
							onClick={() => handleStatusChange(status.id)}>
							<div className="flex justify-between">
								<div
									className={`rounded-full shadow-xs backdrop-blur-xs ${selecttedStatus === status.id ? status.className : status.idleClassName}`}>
									{status.icon}
								</div>
								<div
									className={clsx(
										`${selecttedStatus === status.id ? status.activeClass : status.inActiveClass}`,
										`w-7 h-7 rounded-full flex items-center justify-center`,
									)}
								/>
							</div>
							<div className="flex flex-col">
								<p className="text-3xl font-bold tracking-tight">
									{status.value}
								</p>
								<span
									className={`text-xs font-semibold text-slate-500 uppercase tracking-wider block mb-1 ${selecttedStatus === status.id ? "text-slate-600" : "text-slate-500/50"}`}>
									{status.label}
								</span>
							</div>
						</div>
					))}
				</div>

				<div className="flex flex-col grid-cols-1 gap-6">
					<div className="flex flex-col col-span-2 bg-white border border-slate-100 rounded-4xl p-6 gap-4 shadow-sm">
						<DataTable
							columns={columnsTrainRun(setSelectedRunId)}
							data={filteredData}
							trainRun
						/>
					</div>
					{/* <div className="xl:col-span-1 lg:col-span-2 col-span-1 flex flex-col gap-4">
						{trainingRunsData.map((run) => {
							if (run.runId === selectedRunId) {
								return (
									<DetailTrainRun
										key={run.runId}
										run={run}
									/>
								);
							}
						})}
					</div> */}
				</div>
			</div>

			{/* {modalSft && (
            )} */}
			<ModalRunSFT
				isOpen={modalSft}
				toggleModal={toggleModalSft}
			/>
		</div>
	);
}