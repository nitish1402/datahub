from datahub.ingestion.api.source import Source
from datahub.ingestion.source.sql.oracle import OracleSource, OracleConfig
from datahub.metadata.schema_classes import DatasetSnapshotClass, DatasetPropertiesClass, MetadataChangeEventClass, UpstreamLineageClass, UpstreamClass, DatasetLineageTypeClass, BrowsePathEntryClass, BrowsePathsV2Class, DataPlatformInstanceClass
from datahub.ingestion.api.workunit import MetadataWorkUnit
from sqlalchemy import create_engine, inspect
from datahub.metadata.com.linkedin.pegasus2avro.common import StatusClass
import datahub.emitter.mce_builder as builder
from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.emitter.mcp_builder import (
    add_dataset_to_container,
)
from datahub.ingestion.source.sql.sql_utils import (
    gen_database_key,
    gen_schema_key,
)
from typing import (Iterable, Union)

import sqlparse
import logging

logger = logging.getLogger(__name__)

class OracleStoredProcedureIngestion(OracleSource, Source):

    @classmethod
    def create(cls, config_dict, ctx):
        config = OracleConfig.parse_obj(config_dict)
        return cls(config, ctx)
    
    def make_stored_procedure_with_text(self, conn, proc):
        proc_name = proc[0]
        code_lines = conn.execute("""
                                  SELECT TEXT FROM ALL_SOURCE 
                                  WHERE NAME = :name 
                                  AND OWNER = :owner
                                  AND TYPE IN ('PROCEDURE', 'FUNCTION', 'PACKAGE') 
                                  ORDER BY LINE
                                  """, {'name': proc_name, 'owner': proc[1]})
        code = ''.join(line[0] for line in code_lines)
        return {
            "owner": proc[1],
            "name": proc_name,
            "type": proc[2],
            "text": code,
            "status": proc[3]
        }
                
    def fetch_stored_procedures(self, engine, conn, schema):
        query = """SELECT OBJECT_NAME, OWNER, OBJECT_TYPE, STATUS FROM ALL_OBJECTS WHERE OBJECT_TYPE IN ('PROCEDURE', 'FUNCTION', 'PACKAGE') and owner = :owner"""
        schema_name = engine.dialect.denormalize_name(schema)
        procedures = conn.execute(query, {'owner': schema_name})
        for proc in procedures:
            yield self.make_stored_procedure_with_text(conn, proc)
                
    def fetch_upstream_urns(self, conn, proc_name, owner):
        query = """
                select dep.OWNER, NAME, REFERENCED_OWNER, REFERENCED_NAME, REFERENCED_TYPE, TABLE_OWNER, TABLE_NAME 
                from all_dependencies dep left join all_synonyms syn 
                on dep.REFERENCED_TYPE = 'SYNONYM' and dep.REFERENCED_OWNER = syn.owner and dep.REFERENCED_NAME = syn.SYNONYM_NAME  
                where name = :name 
                and dep.owner = :owner
                and referenced_owner != 'SYS' 
                and (syn.table_owner is null or syn.table_owner != 'SYS')
                """
        dependencies = conn.execute(query, {'name': proc_name, 'owner': owner})
        
        for dependency in dependencies:
            schema_name = dependency[2]
            table_name = dependency[3]
            
            if dependency[4] == 'SYNONYM':
                schema_name = dependency[5]
                table_name = dependency[6]
                
            yield self.datasetUrn(schema_name, table_name)
        
    def datasetUrn(self, schema_name, object_name):
        return builder.make_dataset_urn_with_platform_instance(platform=self.platform, name=self.datasetName(schema_name, object_name), platform_instance = self.config.platform_instance, env=self.config.env)
    
    def datasetName(self, schema_name, object_name):
        return f"{schema_name.lower()}.{object_name.lower()}"
    
    def _process_entity(
        self,
        schema: str,
        db_name: str,
        procedure,
        conn
    ) -> Iterable[Union[MetadataWorkUnit]]:
        urn = self.datasetUrn(procedure['owner'] , procedure['name'])

        schema_container_key = gen_schema_key(
            db_name=db_name,
            schema=schema,
            platform=self.platform,
            platform_instance=self.config.platform_instance,
            env=self.config.env,
        )
        
        yield from add_dataset_to_container(
            container_key=schema_container_key,
            dataset_urn=urn,
        )

        dataset_snapshot = DatasetSnapshotClass(
            urn=urn,
            aspects=[StatusClass(removed=False)],
        )

        dataset_properties = DatasetPropertiesClass(
            name=procedure['name'],
            description=f"{procedure['type']}: {procedure['name']}",
            customProperties={
                    "type": procedure["type"],
                    "owner": procedure["owner"],
                    "text": sqlparse.format(procedure["text"], reindent=True, keyword_case='upper'),
                    "status": procedure["status"]
                },
        )
                        
        upstreams = []
        for upstream_urn in self.fetch_upstream_urns(conn, procedure["name"], procedure["owner"]):
            upstreams.append(UpstreamClass(
                        dataset=upstream_urn,
                        type=DatasetLineageTypeClass.TRANSFORMED,
                    ))
        
        upstream_lineages = UpstreamLineageClass(upstreams=upstreams)
        
        data_platform_aspect = DataPlatformInstanceClass(
                    platform=builder.make_data_platform_urn(self.platform),
                    instance=builder.make_dataplatform_instance_urn(
                        self.platform, self.config.platform_instance
                    )
        )
        print(procedure["name"], len(upstreams))
        dataset_snapshot.aspects.append(dataset_properties)
        dataset_snapshot.aspects.append(upstream_lineages)
        dataset_snapshot.aspects.append(data_platform_aspect)
    
        mce = MetadataChangeEventClass(
            proposedSnapshot = dataset_snapshot
        )
        work_unit = MetadataWorkUnit(id=procedure['name'], mce=mce)
        yield work_unit
        platform_instance_urn = builder.make_dataplatform_instance_urn(
                        self.platform, self.config.platform_instance
                    )
        database_container_key = gen_database_key(
            '',
            platform=self.platform,
            platform_instance=self.config.platform_instance,
            env=self.config.env,
        )
        container_urn = builder.make_container_urn(
            guid=schema_container_key.guid(),
        )
        yield MetadataWorkUnit(
            id=f"{procedure['name']}-paths",
            mcp=MetadataChangeProposalWrapper(
                entityUrn=urn,
                aspect=BrowsePathsV2Class(
                    path=[
                        BrowsePathEntryClass(id=platform_instance_urn, urn=platform_instance_urn),
                        BrowsePathEntryClass(id=database_container_key.as_urn(), urn=database_container_key.as_urn()),
                        BrowsePathEntryClass(id=container_urn, urn=container_urn)
                        ]
                    )
                )
        )
        
        # dataset_snapshot.aspects.append(schema_metadata)

        # yield MetadataWorkUnit(
        #     id=f"{dataset_name}-subtypes",
        #     mcp=MetadataChangeProposalWrapper(
        #         entityUrn=dataset_urn,
        #         aspect=SubTypesClass(typeNames=[DatasetSubTypes.TABLE]),
        #     ),
        # )    
            
    def loop_stored_procedures(  # noqa: C901
        self,
        engine,
        conn,
        db_name: str,
        schema: str,
    ) -> Iterable[Union[MetadataWorkUnit]]:
        try:
            for procedure in self.fetch_stored_procedures(engine, conn, schema):                    
                    # self.report.report_entity_scanned(procedure["name"], ent_type="stored procedure")
                    
                    try:
                        yield from self._process_entity(
                            schema,
                            db_name,
                            procedure,
                            conn,
                        )
                    except Exception as e:
                        self.report.warning(
                            "Error processing entity - stored procedures",
                            context=f"{schema}.{procedure['name']}",
                            exc=e,
                        )
        except Exception as e:
            self.report.failure(
                "Error processing stored procedures",
                context=schema,
                exc=e,
            )
    
    def get_workunits(self):

        # Call the original logic for table ingestion
        yield from super().get_workunits()

        # Add logic for stored procedures
        engine = create_engine(
            self.config.get_sql_alchemy_url()
        )
        
        inspector = inspect(engine)
        db_name = self.get_db_name(inspector)
        # yield from self.gen_database_containers(database=db_name)

        with engine.connect() as conn:
            for schema in self.get_allowed_schemas(inspector, db_name):
                yield from self.loop_stored_procedures(engine=engine, conn=conn, db_name=db_name, schema=schema)
                