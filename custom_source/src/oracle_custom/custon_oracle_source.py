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
            yield inspector

# Use CustomOracleSource instead of OracleSource in your ingestion pipeline