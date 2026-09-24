import {
	CloudIcon,
	NotePencilIcon,
	SpinnerGapIcon,
	StackIcon,
} from "@phosphor-icons/react/dist/ssr";
import { useState } from "react";
import { columnsModelRegistry } from "~/components/table/columns/model-registry-column";
import { DataTable } from "~/components/table/data-table";
import Header from "~/components/ui/header";
import type { ModelRegistryProp } from "~/type";
import { modelRegistryData } from "~/utils";

export default function ModelRegistry() {
	const [selectedStatus, setSelectedStatus] = useState<string>("ALL");
	const [filteredData, setFilteredData] =
		useState<ModelRegistryProp[]>(modelRegistryData);

	const candidateTrain = modelRegistryData.filter(
		(run) => run.status === "candidate",
	);
	const evaluatedTrain = modelRegistryData.filter(
		(run) => run.status === "evaluated",
	);
	const stagingTrain = modelRegistryData.filter(
		(run) => run.status === "staging",
	);
	const productionTrain = modelRegistryData.filter(
		(run) => run.status === "production",
	);
	const archiveTrain = modelRegistryData.filter(
		(run) => run.status === "archive",
	);

	const handleStatusChange = (status: string) => {
		if (status === selectedStatus) {
			setSelectedStatus("ALL");
			setFilteredData(modelRegistryData);
			return;
		}
		setSelectedStatus(status);
		setFilteredData(
			status === "ALL"
				? modelRegistryData
				: modelRegistryData.filter((run) => run.status === status),
		);
	};

	const cardRegistryStatus = [
		{
			label: "Candidate",
			count: candidateTrain.length,
			status: "candidate",
			className:
				"border border-slate-100 text-slate-900 rounded-3xl bg-slate-50/70 shadow-sm",
			idleClassName:
				"border border-slate-100 text-slate-900/50 rounded-3xl bg-slate-50/90 shadow-sm",
			icon: (
				<StackIcon
					className="w-7 h-7"
					weight="bold"
				/>
			),
		},
		{
			label: "Evaluated",
			count: evaluatedTrain.length,
			status: "evaluated",
			className:
				"border border-sky-100 text-sky-900 rounded-3xl bg-sky-50/70 shadow-sm",
			idleClassName:
				"border border-sky-100 text-sky-900/50 rounded-3xl bg-sky-50/90 shadow-sm",
			icon: (
				<NotePencilIcon
					className="w-7 h-7"
					weight="bold"
				/>
			),
		},
		{
			label: "Staging",
			count: stagingTrain.length,
			status: "staging",
			className:
				"border border-amber-100 text-amber-900 rounded-3xl bg-amber-50/70 shadow-sm",
			idleClassName:
				"border border-amber-100 text-amber-900/50 rounded-3xl bg-amber-50/90 shadow-sm",
			icon: (
				<SpinnerGapIcon
					className="w-7 h-7"
					weight="bold"
				/>
			),
		},
		{
			label: "Production",
			count: productionTrain.length,
			status: "production",
			className:
				"border border-emerald-100 text-emerald-900 rounded-3xl bg-emerald-50/70 shadow-sm",
			idleClassName:
				"border border-emerald-100 text-emerald-900/50 rounded-3xl bg-emerald-50/90 shadow-sm",
			icon: (
				<CloudIcon
					className="w-7 h-7"
					weight="bold"
				/>
			),
		},
		// {
		// 	label: "Archive",
		// 	count: archiveTrain.length,
		// 	status: "archive",
		// 	className:
		// 		"border border-rose-100 text-rose-900 rounded-3xl bg-rose-50/70 shadow-sm",
		// 	idleClassName:
		// 		"border border-rose-100 text-rose-900/50 rounded-3xl bg-rose-50/90 shadow-sm",
		// 	icon: (
		// 		<ArchiveIcon
		// 			className="w-7 h-7"
		// 			weight="bold"
		// 		/>
		// 	),
		// },
	];

	return (
		<div>
			<Header title="Model Registry & Lineage" />
			<div className="flex flex-col space-y-4 max-w-7xl w-full mx-auto p-4">
				<div className="grid lg:grid-cols-4 grid-cols-2 gap-4 w-full">
					{cardRegistryStatus.map((status) => (
						<div
							key={status.label}
							className={`${selectedStatus === status.status ? status.className : selectedStatus === "ALL" ? status.className : status.idleClassName} px-5 py-4 flex justify-between transition-all hover:scale-[1.01] h-fit`}
							onClick={() => handleStatusChange(status.status)}>
							<div className="flex flex-col">
								<p className="text-3xl font-bold tracking-tight">
									{status.count}
								</p>
								<span
									className={`text-xs font-semibold text-slate-500 uppercase tracking-wider block mb-1 ${selectedStatus === status.status ? "text-slate-600" : selectedStatus === "ALL" ? "text-slate-600" : "text-slate-500/50"}`}>
									{status.label}
								</span>
							</div>
							<div className="rounded-full shadow-xs backdrop-blur-xs">
								{status.icon}
							</div>
						</div>
					))}
				</div>
				<div></div>
				<div className="p-6 h-full max-w-380 w-full mx-auto shadow-sm rounded-4xl bg-white border border-slate-100 mb-10">
					<DataTable
						columns={columnsModelRegistry()}
						data={filteredData}
						registry
					/>
				</div>
			</div>

			{/* {evaluationOpen && selectedRunId && (
				<ModalEvaluationModel
					isOpen={evaluationOpen}
					toggleClose={toggleClose}
					modelRegist={selectedRunId}
				/>
			)} */}
		</div>
	);
}
