import { useEffect, useState } from "react";
import ModalEvaluationModel from "~/components/modal/model-evaluation";
import { columnsModelRegistry } from "~/components/table/columns/model-registry-column";
import { DataTable } from "~/components/table/data-table";
import Header from "~/components/ui/header";
import type { TrainingRun } from "~/type";
import { trainingRunsData } from "~/utils";

export default function ModelRegistry() {
    const [evaluationOpen, setEvaluationOpen] = useState(false);
    const [modelRegistry, setModelRegistry] = useState<TrainingRun[]>(trainingRunsData);
    const [selectedRunId, setSelectedRunId] = useState<TrainingRun>();

    const toggleOpen = (id: string) => {
        setEvaluationOpen(true);
        const selectedModel = modelRegistry.find((item) => item.runId === id);
        setSelectedRunId(selectedModel);
    }

    const toggleClose = () => {
        setEvaluationOpen((prev) => !prev);
    }

    useEffect(() => {
        const fetchModelRegistry = async () => {
            const newData = trainingRunsData.filter((item) => item.status === "COMPLETED");
            setModelRegistry(newData);
        }
        fetchModelRegistry();
    }, [])

    return (
        <div>
            <Header title="Model Registry & Lineage" />
            <div className="pl-4 pr-4 pb-4 h-full max-w-380 w-full mx-auto">
                <DataTable columns={columnsModelRegistry(toggleOpen)} data={trainingRunsData} registry />
            </div>

            {evaluationOpen && selectedRunId && (
                <ModalEvaluationModel isOpen={evaluationOpen} toggleClose={toggleClose} modelRegist={selectedRunId} />
            )}
        </div>
    )
}