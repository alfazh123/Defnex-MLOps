import { useState } from "react";

import DetailTrainRun from "~/components/detail-train-run";
import ModalRunSFT from "~/components/modal/run-sft";
import { columnsTrainRun } from "~/components/table/columns/train-column";
import { DataTable } from "~/components/table/data-table";
import { Button } from "~/components/ui/button";
import Header from "~/components/ui/header";

import { trainingRunsData } from "~/utils";

import { Circle, CircleCheck, RefreshCcw } from "lucide-react";

export default function TrainRun() {
    const [modalSft, setModalSft] = useState(false);
    const [selectedRunId, setSelectedRunId] = useState<string>(trainingRunsData[0]?.runId || "");

    const toggleModalSft = () => {
        setModalSft((prevState) => !prevState);
    }

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

	const cardTrainStatus = [
		{
			label: "Completed",
			value: completedTrain.length,
			className:
				"border border-green-500 text-green-500 rounded-md bg-green-50",
			icon: <CircleCheck className="w-6 h-6" />,
		},
		{
			label: "Failed",
			value: failedTrain.length,
			className:
				"border border-red-500 text-red-500 rounded-md bg-red-50",
			icon: <Circle className="w-6 h-6" />,
		},
		{
			label: "Running",
			value: runningTrain.length,
			className:
				"border border-blue-500 text-blue-500 rounded-md bg-blue-50",
			icon: <Circle className="w-6 h-6" />,
		},
		{
			label: "Pending",
			value: pendingTrain.length,
			className:
				"border border-yellow-500 text-yellow-500 rounded-md bg-yellow-50",
			icon: <Circle className="w-6 h-6" />,
		},
	];

    return (
		<div className="flex flex-col h-screen">
			<Header
				title="Training and Run"
				sft
				modalSft={toggleModalSft}
			/>
			<div className="px-4 pb-20 h-full max-w-380 w-full mx-auto">
				<div className="flex justify-between items-center mb-4">
					<div>
						<h3 className="text-lg font-semibold">
							SFT Execution & Training Runs
						</h3>
						<p className="text-sm text-accent-foreground">
							Unsloth-backed Supervised Fine-Tuning jobs and mock
							worker tracking
						</p>
					</div>
					<div>
						<Button variant="outline">
							<RefreshCcw className="mr-2 h-4 w-4" />
							Refresh worker
						</Button>
					</div>
				</div>

				<div className="grid md:grid-cols-4 grid-cols-2 my-4 gap-4">
					{cardTrainStatus.map((status) => (
						<div
							key={status.label}
							className={`${status.className} px-4 py-2 flex  justify-between items-center`}>
							<div>
								<span>{status.label}</span>
								<p className="text-lg font-bold">
									{status.value}
								</p>
							</div>
							{status.icon}
						</div>
					))}
				</div>

				<div className="flex flex-col grid-cols-1 gap-4">
					<div className="flex flex-col col-span-2 bg-white border rounded-md p-4 gap-4">
						<DataTable
							columns={columnsTrainRun(setSelectedRunId)}
							data={trainingRunsData}
							trainRun
						/>
					</div>
					<div className="xl:col-span-1 lg:col-span-2 col-span-1 flex flex-col gap-4">
						{trainingRunsData.map((run) => {
							if (run.runId === selectedRunId) {
								return <DetailTrainRun run={run} />;
							}
						})}
					</div>
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