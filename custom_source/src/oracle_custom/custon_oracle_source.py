from typing import Any, Dict, Iterable, List
from datahub.ingestion.source.sql.oracle import OracleSource
from sqlalchemy.engine.reflection import Inspector

class CustomOracleSource(OracleSource):
    def get_stored_procedures(self, inspector: Inspector) -> List[Dict[str, Any]]:
        query = """
        SELECT object_name, object_type, status
        FROM all_objects
        WHERE object_type IN ('PROCEDURE', 'FUNCTION')
        """
        result = inspector.bind.execute(query)
        procedures = []
        for row in result:
            procedures.append({
                "name": row["object_name"],
                "type": row["object_type"],
                "status": row["status"]
            })
        return procedures

    def get_inspectors(self) -> Iterable[Inspector]:
        for inspector in super().get_inspectors():
            procedures = self.get_stored_procedures(inspector)
            for procedure in procedures:
                print(f"Found procedure: {procedure['name']} of type {procedure['type']}")
            
            for proc in procedures:
                dataset_urn = make_dataset_urn(platform="oracle", name=f"{proc['schema']}.{proc['name']}", env="PROD")
                dataset_snapshot = DatasetSnapshotClass(
                    urn=dataset_urn,
                    aspects=[],
                )
                # Add description or other metadata
                dataset_snapshot.aspects.append({"description": proc["description"]})

                # Create a MetadataChangeEvent
                mce = MetadataChangeEventClass(proposedSnapshot=dataset_snapshot)

                # Emit the MCE to DataHub
                emitter.emit_mce(mce)
            
            yield inspector

# Use CustomOracleSource instead of OracleSource in your ingestion pipeline