from operator import inv
import requests
from pydantic import BaseModel
from bs4 import BeautifulSoup
import json
import random
from agent.tools import Tool
from typing import Union, Dict
from typing import List, Optional, Union


class ChemicalPropAPI:
    def __init__(self) -> None:
        self._endpoint = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"

    def get_name_by_cid(self, cid: str, top_k: Optional[int] = None) -> List[str]:
        html_doc = requests.get(f"{self._endpoint}cid/{cid}/synonyms/XML").text
        soup = BeautifulSoup(html_doc, "html.parser", from_encoding="utf-8")
        syns = soup.find_all('synonym')
        ans = []
        if top_k is None:
            top_k = len(syns)
        for syn in syns[:top_k]:
            ans.append(syn.text)
        return ans

    def get_cid_by_struct(self, smiles: str) -> List[str]:
        html_doc = requests.get(
            f"{self._endpoint}smiles/{smiles}/cids/XML").text
        soup = BeautifulSoup(html_doc, "html.parser", from_encoding="utf-8")
        cids = soup.find_all('cid')
        if cids is None:
            return []
        ans = []
        for cid in cids:
            ans.append(cid.text)
        return ans

    def get_cid_by_name(self, name: str, name_type: Optional[str] = None) -> List[str]:
        url = f"{self._endpoint}name/{name}/cids/XML"
        if name_type is not None:
            url += f"?name_type={name_type}"
        html_doc = requests.get(url).text
        soup = BeautifulSoup(html_doc, "html.parser", from_encoding="utf-8")
        cids = soup.find_all('cid')
        if cids is None:
            return []
        ans = []
        for cid in cids:
            ans.append(cid.text)
        return ans

    def get_prop_by_cid(self, cid: str) -> str:
        html_doc = requests.get(f"{self._endpoint}cid/{cid}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,IUPACName,XLogP,ExactMass,MonoisotopicMass,TPSA,Complexity,Charge,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,HeavyAtomCount,CovalentUnitCount/json").text
        return json.loads(html_doc)['PropertyTable']['Properties'][0]


class GetNameResponse(BaseModel):

    """name list"""
    names: List[str]


class GetStructureResponse(BaseModel):

    """structure list"""
    state: int
    content: Optional[str] = None


class GetIDResponse(BaseModel):
    state: int
    content: Union[str, List[str]]


chemical_prop_api = ChemicalPropAPI()


def get_name(cid: str):
    """prints the possible 3 synonyms of the queried compound ID"""
    ans = chemical_prop_api.get_name_by_cid(cid, top_k=3)
    return {
        "names": ans
    }


class GetNameTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "GetName"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        cid = invoke_data
        return get_name(cid), 0, False, {}

    def description(self) -> str:
        return "GetName(cid), prints the possible 3 synonyms of the queried compound ID."


class GetAllNameTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "GetAllName"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        cid = invoke_data
        ans = chemical_prop_api.get_name_by_cid(cid)
        return {
            "names": ans
        }, 0, False, {}

    def description(self) -> str:
        return "GetAllName(cid), prints all the possible synonyms (might be too many, use this function carefully)."


class GetIdByStructTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "GetIdByStruct"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        smiles = invoke_data
        cids = chemical_prop_api.get_cid_by_struct(smiles)
        if len(cids) == 0:
            return {
                "state": "no result"
            }, 0, False, {}
        else:
            return {
                "state": "matched",
                "content": cids[0]
            }, 0, False, {}

    def description(self) -> str:
        return "GetIdByStruct(smiles), prints the ID of the queried compound SMILES. This should only be used if smiles is provided or retrieved in the previous step. The input should not be a string, but a SMILES formula."


class GetIdTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "GetId"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        name = invoke_data
        cids = chemical_prop_api.get_cid_by_name(name)
        if len(cids) > 0:
            return {
                "state": "precise",
                "content": cids[0]
            }, 0, False, {}

        cids = chemical_prop_api.get_cid_by_name(name, name_type="word")
        if len(cids) > 0:
            if name in get_name(cids[0]):
                return {
                    "state": "precise",
                    "content": cids[0]
                }, 0, False, {}

        ans = []
        random.shuffle(cids)
        for cid in cids[:5]:
            nms = get_name(cid)
            ans.append(nms)
        return {
            "state": "not precise",
            "content": ans
        }, 0, False, {}

    def description(self) -> str:
        return "GetId(name), prints the ID of the queried compound name, and prints the possible 5 names if the queried name can not been precisely matched."


class GetPropTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "GetProp"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        cid = invoke_data
        return chemical_prop_api.get_prop_by_cid(cid), 0, False, {}

    def description(self) -> str:
        return "GetProp(cid), prints the properties of the queried compound ID."
