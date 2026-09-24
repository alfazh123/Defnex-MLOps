import { useState } from "react";
import Layout from "~/components/layout";
import Header from "~/components/ui/header";
import AddDatasetModal from "~/components/modal/add-dataset";
import { datasets } from "~/utils";
import { DatasetTable } from "~/components/table/dataset-table";
import { columns } from "~/components/table/columns/dataset-column";
import { ScrollArea } from "~/components/ui/scroll-area";
import {
	ArrowArcRightIcon,
	ArrowRightIcon,
	CheckCircleIcon,
	QueueIcon,
	SneakerMoveIcon,
	XCircleIcon,
} from "@phosphor-icons/react/dist/ssr";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import clsx from "clsx";

export default function NewKnowledge() {
    const [modalAddKnwledge, setModalAddKnowledge] = useState(false);

    const toggleModalAddKnowledge = () => {
        setModalAddKnowledge((prevState) => !prevState);
        console.log("modalAddKnwledge", modalAddKnwledge);
    }

	const [selectedStatus, setSelectedStatus] = useState<string>("");
	const [filteredData, setFilteredData] = useState(datasets);

	const handleStatusChange = (status: string) => {
		if (status === selectedStatus) {
			setSelectedStatus("");
			setFilteredData(datasets);
			return;
		}
		setSelectedStatus(status);
		setFilteredData(
			status === ""
				? datasets
				: datasets.filter((dataset) => dataset.sftStatus === status),
		);
	};

	const statusBadges = [
		// { id: "ALL", label: "All", className: "bg-slate-50 text-slate-600 border border-slate-200/80", idleClassName: "bg-slate-50/90 text-slate-600/50 border border-slate-200/60" },
		{
			id: "Ready",
			label: "Ready",
			length: datasets.filter((dataset) => dataset.sftStatus === "Ready")
				.length,
			icon: <CheckCircleIcon className="w-4 h-4" />,
			className:
				"bg-emerald-50 text-emerald-600 border border-emerald-200/80",
			idleClassName:
				"bg-emerald-50/90 text-emerald-600/50 border border-emerald-200/60",
		},
		{
			id: "Training",
			label: "Training",
			length: datasets.filter(
				(dataset) => dataset.sftStatus === "Training",
			).length,
			icon: <SneakerMoveIcon className="w-4 h-4" />,
			className: "bg-amber-50 text-amber-600 border border-amber-200/80",
			idleClassName:
				"bg-amber-50/90 text-amber-600/50 border border-amber-200/60",
		},
		{
			id: "Queued",
			label: "Queued",
			length: datasets.filter((dataset) => dataset.sftStatus === "Queued")
				.length,
			icon: <QueueIcon className="w-4 h-4" />,
			className: "bg-sky-50 text-sky-600 border border-sky-200/80",
			idleClassName:
				"bg-sky-50/90 text-sky-600/50 border border-sky-200/60",
		},
		{
			id: "Failed",
			label: "Failed",
			length: datasets.filter((dataset) => dataset.sftStatus === "Failed")
				.length,
			icon: <XCircleIcon className="w-4 h-4" />,
			className: "bg-rose-50 text-rose-600 border border-rose-200/80",
			idleClassName:
				"bg-rose-50/90 text-rose-600/50 border border-rose-200/60",
		},
	];

    return (
		<>
			{/* <div className="w-full h-full flex flex-col overflow-y-auto bg-sidebar rounded-lg"> */}
			<Header
				title="Datasets & Intake Manifests"
				description="Manage versioned training datasets and schema manifests"
				newKnowledge
				modalAddKnowledge={toggleModalAddKnowledge}
			/>
			<ScrollArea className="flex flex-col gap-4 rounded-2xl">
				{/* Cards Grid */}
				<div className="flex flex-col gap-4 p-8 max-w-7xl w-full mx-auto shadow-sm border rounded-4xl">
					<div className="flex justify-between">
						<div className="flex flex-col gap-2">
							<Label
								htmlFor="search"
								className="text-sm font-semibold text-slate-600">
								Search Datasets
							</Label>
							<Input
								id="search"
								placeholder="Search datasets..."
								className="w-full max-w-sm"
							/>
						</div>
						<div className="flex gap-2 items-center">
							{statusBadges.map((badge) => (
								<div
									key={badge.id}
									onClick={() => handleStatusChange(badge.id)}
									className={clsx(
										"px-3 py-1.5 flex gap-1 rounded-full text-xs font-semibold transition-all cursor-pointer",
										selectedStatus === badge.id
											? badge.className
											: badge.idleClassName,
									)}>
									{badge.icon} {badge.length} {badge.label}
								</div>
							))}
						</div>
					</div>
					<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
						{filteredData.map((item) => (
							<div
								key={item.id}
								className="bg-white rounded-2xl border border-slate-200/80 shadow-xs flex flex-col justify-between hover:border-slate-300 hover:scale-105 transition-all">
								<div
									className={clsx(
										"p-4 rounded-xl m-1",
										item.sftStatus === "Ready" &&
											"text-emerald-600 bg-emerald-50",
										item.sftStatus === "Training" &&
											"text-amber-600 bg-amber-50",
										item.sftStatus === "Queued" &&
											"text-sky-600 bg-sky-50",
										item.sftStatus === "Failed" &&
											"text-rose-600 bg-rose-50",
									)}>
									{/* Card Header: Title & Status Badge */}
									<div
										className={clsx(
											"flex justify-between items-start mb-4 gap-2 border-b border-slate-200 pb-2",
										)}>
										{item.sftStatus === "Ready" && (
											<CheckCircleIcon className="w-5 h-5" />
										)}
										{item.sftStatus === "Training" && (
											<SneakerMoveIcon className="w-5 h-5" />
										)}
										{item.sftStatus === "Queued" && (
											<QueueIcon className="w-5 h-5" />
										)}
										{item.sftStatus === "Failed" && (
											<XCircleIcon className="w-5 h-5" />
										)}
										<h4
											className={clsx(
												"font-bold text-sm tracking-tight truncate",
												item.sftStatus === "Ready" &&
													"text-emerald-600",
												item.sftStatus === "Training" &&
													"text-amber-600",
												item.sftStatus === "Queued" &&
													"text-sky-600",
												item.sftStatus === "Failed" &&
													"text-rose-600",
											)}>
											{item.title}
										</h4>
										{/* <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-600 border border-emerald-200/60 text-xs font-semibold shrink-0">
											<CheckCircleIcon className="w-3.5 h-3.5" />
											<span>{item.sftStatus}</span>
										</div> */}
									</div>

									{/* Card Info Rows */}
									<div className="space-y-1.5 text-xs">
										<div className="flex justify-between items-center">
											<span className="text-slate-700">
												Samples:
											</span>
											<span className="font-bold text-slate-900">
												{item.totalRows.toLocaleString()}
											</span>
										</div>
										<div className="flex justify-between items-center">
											<span className="text-slate-700">
												Format:
											</span>
											<span className="text-slate-600 font-medium lower">
												{item.format.toLowerCase()}
											</span>
										</div>
										<div className="pt-2">
											<p className="text-[11px] text-slate-700 truncate">
												s3://defnex-mlops/datasets/
												{item.fileName}
											</p>
										</div>
									</div>
								</div>

								{/* Card Footer Link */}
								<a
									className="group p-5 flex items-center justify-between text-xs font-semibold cursor-pointer transition-colors"
									href={`/dataset/${item.id}`}>
									<span>Inspect Manifest</span>
									<ArrowRightIcon className="w-4 h-4 transition-transform group-hover:translate-x-1 -rotate-45 group-hover:rotate-0" />
								</a>
							</div>
						))}
					</div>
					{/* <div className="flex-1 h-full p-4">
						<DatasetTable
							columns={columns}
							data={datasets}
						/>
					</div> */}
				</div>
			</ScrollArea>

			{/* {modalAddKnwledge && (
                )} */}
			<AddDatasetModal
				isOpen={modalAddKnwledge}
				toggleModal={toggleModalAddKnowledge}
			/>
			{/* </div> */}
		</>
	);
}